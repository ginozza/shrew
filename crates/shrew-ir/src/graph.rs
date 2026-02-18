// Graph IR — Validated intermediate representation for .sw programs
//
// The Graph IR is the layer between the raw AST and execution. It represents
// the computation as a directed acyclic graph (DAG) where:
//
//   - Each node is a well-typed operation (matmul, add, relu, etc.)
//   - Edges represent data flow between operations
//   - Parameters and inputs are explicitly tracked
//   - The graph is validated: no dangling references, type-checked dims
//
// The IR is designed for:
//   1. Validation — catch errors before execution
//   2. Optimization — constant folding, fusion, dead code elimination
//   3. Scheduling — determine execution order
//   4. Code generation — emit backend-specific kernels
//
// ARCHITECTURE:
//   AST (from parser) ► Lowering ► GraphIR (this module)
//                                         │
//                                         ├ validate()
//                                         ├ optimize()
//                                         └ schedule() → execution plan

use std::collections::HashMap;
use std::fmt;

// Node identifiers

/// Unique identifier for a node in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

// Type information

/// A resolved tensor shape — dimensions are either concrete or symbolic.
#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    /// Known at compile time: 768, 50257
    Fixed(i64),
    /// Symbolic, resolved at runtime: Batch, SeqLen
    Symbolic(String),
    /// Unknown / dynamic
    Dynamic,
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Fixed(n) => write!(f, "{n}"),
            Dim::Symbolic(s) => write!(f, "{s}"),
            Dim::Dynamic => write!(f, "?"),
        }
    }
}

/// Resolved data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F16,
    F32,
    F64,
    Bf16,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
    Complex64,
    Complex128,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F16 => write!(f, "f16"),
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::Bf16 => write!(f, "bf16"),
            DType::I8 => write!(f, "i8"),
            DType::I16 => write!(f, "i16"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::U8 => write!(f, "u8"),
            DType::U16 => write!(f, "u16"),
            DType::U32 => write!(f, "u32"),
            DType::U64 => write!(f, "u64"),
            DType::Bool => write!(f, "bool"),
            DType::Complex64 => write!(f, "complex64"),
            DType::Complex128 => write!(f, "complex128"),
        }
    }
}

/// The resolved type of a value in the graph.
#[derive(Debug, Clone, PartialEq)]
pub enum IrType {
    /// A tensor with shape and dtype.
    Tensor { shape: Vec<Dim>, dtype: DType },
    /// A scalar value.
    Scalar(DType),
    /// Integer (used for things like dimension values).
    Int,
    /// String (used for attribute values).
    Str,
    /// Boolean.
    Boolean,
    /// Unknown / to be inferred.
    Unknown,
}

impl fmt::Display for IrType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrType::Tensor { shape, dtype } => {
                let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                write!(f, "Tensor<[{}], {}>", dims.join(", "), dtype)
            }
            IrType::Scalar(dt) => write!(f, "{dt}"),
            IrType::Int => write!(f, "int"),
            IrType::Str => write!(f, "str"),
            IrType::Boolean => write!(f, "bool"),
            IrType::Unknown => write!(f, "?"),
        }
    }
}

// Operations

/// An operation in the computation graph.
#[derive(Debug, Clone)]
pub enum OpKind {
    // Tensor creation
    /// Look up rows in an embedding table.
    Embedding,
    /// Generate a range of values.
    Range,

    // Unary ops
    Neg,
    Relu,
    Gelu,
    Silu,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Sqrt,
    Transpose,

    // Binary ops
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    MatMul,

    // Reduction ops
    Sum {
        dims: Vec<i64>,
        keepdim: bool,
    },
    Mean {
        dims: Vec<i64>,
        keepdim: bool,
    },
    Max {
        dim: i64,
        keepdim: bool,
    },
    Min {
        dim: i64,
        keepdim: bool,
    },
    Variance {
        dims: Vec<i64>,
        keepdim: bool,
    },

    //  Normalization 
    LayerNorm {
        eps: f64,
    },
    BatchNorm {
        eps: f64,
    },

    //  Attention 
    MultiHeadAttention {
        n_heads: i64,
    },
    TransformerBlock {
        n_heads: i64,
    },
    Softmax {
        dim: i64,
    },

    //  Shape ops 
    Reshape {
        target_shape: Vec<Dim>,
    },
    View {
        target_shape: Vec<Dim>,
    },
    Permute {
        dims: Vec<i64>,
    },
    Concat {
        dim: i64,
    },
    Split {
        dim: i64,
        chunks: i64,
    },
    Expand {
        target_shape: Vec<Dim>,
    },

    //  Dropout 
    Dropout {
        p: f64,
    },

    //  Linear 
    Linear {
        bias: bool,
    },

    //  Loss functions 
    CrossEntropy,
    MseLoss,

    //  Comparison 
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,

    //  Logical 
    And,
    Or,
    Not,

    //  Constants 
    Constant(ConstantValue),

    //  Control flow 
    Repeat {
        count: i64,
        body_op: Box<OpKind>,
    },

    //  Custom / user-defined 
    Custom {
        name: String,
        attrs: HashMap<String, AttrValue>,
    },

    //  Graph call (calling another @graph) 
    Call {
        graph_name: String,
    },

    //  Identity (pass-through, used for inputs) 
    Identity,
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpKind::Add => write!(f, "add"),
            OpKind::Sub => write!(f, "sub"),
            OpKind::Mul => write!(f, "mul"),
            OpKind::Div => write!(f, "div"),
            OpKind::MatMul => write!(f, "matmul"),
            OpKind::Embedding => write!(f, "embedding"),
            OpKind::LayerNorm { eps } => write!(f, "layer_norm(eps={eps})"),
            OpKind::Softmax { dim } => write!(f, "softmax(dim={dim})"),
            OpKind::Relu => write!(f, "relu"),
            OpKind::Gelu => write!(f, "gelu"),
            OpKind::Transpose => write!(f, "transpose"),
            OpKind::Constant(v) => write!(f, "const({v})"),
            OpKind::Custom { name, .. } => write!(f, "custom({name})"),
            OpKind::Call { graph_name } => write!(f, "call({graph_name})"),
            OpKind::Identity => write!(f, "identity"),
            other => write!(f, "{other:?}"),
        }
    }
}

/// A constant value embedded in the graph.
#[derive(Debug, Clone)]
pub enum ConstantValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Null,
}

impl fmt::Display for ConstantValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstantValue::Int(n) => write!(f, "{n}"),
            ConstantValue::Float(v) => write!(f, "{v}"),
            ConstantValue::Str(s) => write!(f, "\"{s}\""),
            ConstantValue::Bool(b) => write!(f, "{b}"),
            ConstantValue::Null => write!(f, "null"),
        }
    }
}

/// An attribute value (for custom ops and node attrs).
#[derive(Debug, Clone)]
pub enum AttrValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    List(Vec<AttrValue>),
}

// Graph nodes

/// A node in the computation graph.
#[derive(Debug, Clone)]
pub struct IrNode {
    /// Unique identifier.
    pub id: NodeId,
    /// User-visible name (from the .sw source).
    pub name: String,
    /// The operation this node performs.
    pub op: OpKind,
    /// Input edges: which nodes feed into this one.
    pub inputs: Vec<NodeId>,
    /// The resolved type of this node's output.
    pub output_type: IrType,
    /// Optional attributes (key-value metadata).
    pub attrs: HashMap<String, AttrValue>,
    /// Execution hints from the source.
    pub hints: Vec<IrHint>,
}

/// Execution hints attached to nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrHint {
    RecomputeInBackward,
    MustPreserve,
    InPlace,
    NoGrad,
    Custom(String),
}

// Parameters (learnable weights)

/// A learnable parameter in the graph.
#[derive(Debug, Clone)]
pub struct IrParam {
    /// Reference to the node that holds the param value.
    pub node_id: NodeId,
    /// Parameter name.
    pub name: String,
    /// Type (always a Tensor).
    pub ty: IrType,
    /// Initialization strategy.
    pub init: InitStrategy,
    /// Whether the parameter is frozen (no gradient).
    pub frozen: bool,
}

/// How to initialize a parameter.
#[derive(Debug, Clone)]
pub enum InitStrategy {
    Zeros,
    Ones,
    Normal { mean: f64, std: f64 },
    Uniform { low: f64, high: f64 },
    XavierUniform,
    XavierNormal,
    KaimingUniform,
    KaimingNormal,
    Custom(String),
}

// Assertions

/// A compile-time or runtime assertion from @assert.
#[derive(Debug, Clone)]
pub struct IrAssert {
    /// Human-readable description.
    pub message: Option<String>,
    /// The assertion expression as a string (for diagnostics).
    pub expr_text: String,
}

// Graph definition

/// An output of a graph, preserving the user-facing name across optimisations.
#[derive(Debug, Clone)]
pub struct IrOutput {
    /// The user-visible output name (e.g. "out" from `output out;`).
    pub name: String,
    /// The node that produces this output (may be remapped by optimisations).
    pub node_id: NodeId,
}

#[derive(Debug, Clone)]
pub struct IrGraph {
    /// Graph name (e.g., "Forward").
    pub name: String,
    /// All nodes, indexed by NodeId.
    pub nodes: Vec<IrNode>,
    /// Input nodes (by NodeId).
    pub inputs: Vec<NodeId>,
    /// Named outputs.
    pub outputs: Vec<IrOutput>,
    /// Learnable parameters.
    pub params: Vec<IrParam>,
    /// Assertions to verify.
    pub asserts: Vec<IrAssert>,
    /// Node lookup by name.
    pub name_to_id: HashMap<String, NodeId>,
}

impl IrGraph {
    /// Create a new empty graph.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            params: Vec::new(),
            asserts: Vec::new(),
            name_to_id: HashMap::new(),
        }
    }

    /// Add a node and return its NodeId.
    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        op: OpKind,
        inputs: Vec<NodeId>,
        output_type: IrType,
    ) -> NodeId {
        let id = NodeId(self.nodes.len());
        let name = name.into();
        self.name_to_id.insert(name.clone(), id);
        self.nodes.push(IrNode {
            id,
            name,
            op,
            inputs,
            output_type,
            attrs: HashMap::new(),
            hints: Vec::new(),
        });
        id
    }

    /// Register a node as an output. Uses the node's name as the output name.
    pub fn add_output(&mut self, node_id: NodeId) {
        let name = self.nodes[node_id.0].name.clone();
        self.outputs.push(IrOutput { name, node_id });
    }

    /// Register a node as an output with a custom name.
    pub fn add_output_named(&mut self, name: impl Into<String>, node_id: NodeId) {
        self.outputs.push(IrOutput {
            name: name.into(),
            node_id,
        });
    }

    /// Look up a node by name.
    pub fn get_node(&self, name: &str) -> Option<&IrNode> {
        self.name_to_id.get(name).map(|id| &self.nodes[id.0])
    }

    /// Get a node by its ID.
    pub fn node(&self, id: NodeId) -> &IrNode {
        &self.nodes[id.0]
    }

    /// Get a mutable reference to a node by its ID.
    pub fn node_mut(&mut self, id: NodeId) -> &mut IrNode {
        &mut self.nodes[id.0]
    }

    /// Return the total number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Return a topological ordering of node IDs for execution.
    pub fn topo_order(&self) -> Vec<NodeId> {
        let n = self.nodes.len();
        let mut in_degree = vec![0u32; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for node in &self.nodes {
            for &inp in &node.inputs {
                adj[inp.0].push(node.id.0);
                in_degree[node.id.0] += 1;
            }
        }

        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(u) = queue.pop() {
            order.push(NodeId(u));
            for &v in &adj[u] {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push(v);
                }
            }
        }

        order
    }

    /// Pretty-print the graph for debugging.
    pub fn dump(&self) -> String {
        let mut out = format!(
            "=== IrGraph: {} ({} nodes) ===\n",
            self.name,
            self.nodes.len()
        );

        for node in &self.nodes {
            let inputs: Vec<String> = node
                .inputs
                .iter()
                .map(|id| format!("{}({})", self.nodes[id.0].name, id))
                .collect();
            out.push_str(&format!(
                "  {} [{}]: {} <- [{}] :: {}\n",
                node.id,
                node.name,
                node.op,
                inputs.join(", "),
                node.output_type,
            ));
            for hint in &node.hints {
                out.push_str(&format!("    hint: {hint:?}\n"));
            }
        }

        out.push_str(&format!("  inputs:  {:?}\n", self.inputs));
        out.push_str(&format!(
            "  outputs: {:?}\n",
            self.outputs
                .iter()
                .map(|o| (&o.name, o.node_id))
                .collect::<Vec<_>>()
        ));
        out.push_str(&format!("  params:  {} total\n", self.params.len()));
        out
    }
}

// Program-level IR (multiple graphs + config)

/// The full program IR — lowered from the AST.
#[derive(Debug, Clone)]
pub struct IrProgram {
    /// Model metadata.
    pub metadata: HashMap<String, String>,
    /// Configuration values.
    pub config: HashMap<String, ConfigValue>,
    /// Named type aliases.
    pub type_aliases: HashMap<String, IrType>,
    /// Computation graphs.
    pub graphs: Vec<IrGraph>,
    /// Training configuration.
    pub training: Option<TrainingConfig>,
    /// Inference configuration.
    pub inference: Option<InferenceConfig>,
}

impl IrProgram {
    pub fn new() -> Self {
        Self {
            metadata: HashMap::new(),
            config: HashMap::new(),
            type_aliases: HashMap::new(),
            graphs: Vec::new(),
            training: None,
            inference: None,
        }
    }

    /// Find a graph by name.
    pub fn get_graph(&self, name: &str) -> Option<&IrGraph> {
        self.graphs.iter().find(|g| g.name == name)
    }
}

impl Default for IrProgram {
    fn default() -> Self {
        Self::new()
    }
}

/// A configuration value.
#[derive(Debug, Clone)]
pub enum ConfigValue {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    List(Vec<ConfigValue>),
}

impl fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigValue::Int(n) => write!(f, "{n}"),
            ConfigValue::Float(v) => write!(f, "{v}"),
            ConfigValue::Str(s) => write!(f, "\"{s}\""),
            ConfigValue::Bool(b) => write!(f, "{b}"),
            ConfigValue::List(items) => {
                let s: Vec<String> = items.iter().map(|i| i.to_string()).collect();
                write!(f, "[{}]", s.join(", "))
            }
        }
    }
}

/// Training configuration (lowered from @training).
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub model_graph: String,
    pub loss: String,
    pub optimizer: OptimizerConfig,
    pub lr_schedule: Option<LrScheduleConfig>,
    pub grad_clip: Option<GradClipConfig>,
    pub precision: String,
    pub epochs: i64,
    pub batch_size: i64,
    pub accumulation_steps: i64,
}

#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub kind: String,
    pub lr: f64,
    pub extra: HashMap<String, ConfigValue>,
}

#[derive(Debug, Clone)]
pub struct LrScheduleConfig {
    pub kind: String,
    pub extra: HashMap<String, ConfigValue>,
}

#[derive(Debug, Clone)]
pub struct GradClipConfig {
    pub kind: String,
    pub extra: HashMap<String, ConfigValue>,
}

/// Inference configuration (lowered from @inference).
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub model_graph: String,
    pub quantization: Option<HashMap<String, ConfigValue>>,
    pub generation: Option<HashMap<String, ConfigValue>>,
}
