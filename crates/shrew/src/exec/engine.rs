// Engine — Core graph execution engine
//
// Walks the IrGraph in topological order, dispatching each node to the
// appropriate tensor operation. Manages parameter initialization and the
// mapping from NodeId → live Tensor.

use std::collections::HashMap;

use shrew_core::backend::Backend;
use shrew_core::dtype::DType as CoreDType;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use shrew_ir::graph::{
    ConfigValue, ConstantValue, DType as IrDType, Dim, InitStrategy, IrGraph, IrNode, IrProgram,
    IrType, OpKind,
};

use shrew_nn::{
    cross_entropy_loss, mse_loss, Dropout, Embedding, LayerNorm, Linear, Module, TransformerBlock,
};

// Runtime configuration

/// Runtime configuration for resolving symbolic dimensions and execution mode.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Maps symbolic dimension names to concrete values (e.g., "Batch" → 4).
    pub dims: HashMap<String, usize>,
    /// Default data type when unspecified (default: F32).
    pub default_dtype: CoreDType,
    /// Whether we're in training mode (affects dropout, etc.).
    pub training: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            dims: HashMap::new(),
            default_dtype: CoreDType::F32,
            training: false,
        }
    }
}

impl RuntimeConfig {
    /// Set a symbolic dimension value.
    pub fn set_dim(mut self, name: impl Into<String>, value: usize) -> Self {
        self.dims.insert(name.into(), value);
        self
    }

    /// Set training mode.
    pub fn with_training(mut self, training: bool) -> Self {
        self.training = training;
        self
    }

    /// Set default dtype.
    pub fn with_dtype(mut self, dtype: CoreDType) -> Self {
        self.default_dtype = dtype;
        self
    }
}

// Execution result

/// The result of executing a graph.
#[derive(Debug)]
pub struct ExecResult<B: Backend> {
    /// Output tensors, keyed by node name.
    pub outputs: HashMap<String, Tensor<B>>,
    /// All intermediate values, keyed by NodeId.
    pub values: HashMap<usize, Tensor<B>>,
}

impl<B: Backend> ExecResult<B> {
    /// Get the first (or only) output tensor.
    pub fn output(&self) -> Option<&Tensor<B>> {
        self.outputs.values().next()
    }

    /// Get an output by node name.
    pub fn get(&self, name: &str) -> Option<&Tensor<B>> {
        self.outputs.get(name)
    }
}

// Executor

/// Executes IrProgram graphs on the Shrew tensor runtime.
pub struct Executor<B: Backend> {
    /// The lowered IR program.
    program: IrProgram,
    /// Runtime configuration (symbolic dims, dtype, training mode).
    config: RuntimeConfig,
    /// Device to execute on.
    device: B::Device,
    /// Initialized parameter tensors, keyed by (graph_name, param_name).
    params: HashMap<(String, String), Tensor<B>>,
}

impl<B: Backend> Executor<B> {
    /// Create a new executor. Initializes all parameters.
    pub fn new(program: IrProgram, device: B::Device, config: RuntimeConfig) -> Result<Self> {
        let mut exec = Self {
            program,
            config,
            device,
            params: HashMap::new(),
        };
        exec.init_all_params()?;
        Ok(exec)
    }

    /// Get the underlying IR program.
    pub fn program(&self) -> &IrProgram {
        &self.program
    }

    /// Get a reference to the runtime config.
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Get a mutable reference to the runtime config.
    pub fn config_mut(&mut self) -> &mut RuntimeConfig {
        &mut self.config
    }

    /// Get all parameter tensors.
    pub fn params(&self) -> &HashMap<(String, String), Tensor<B>> {
        &self.params
    }

    /// Get flattened parameter list (all graphs).
    pub fn all_params(&self) -> Vec<Tensor<B>> {
        self.params.values().cloned().collect()
    }

    /// Get all parameters as `(key, tensor)` pairs, where key = `"graph/param"`.
    pub fn named_params(&self) -> Vec<(String, Tensor<B>)> {
        let mut pairs: Vec<(String, Tensor<B>)> = self
            .params
            .iter()
            .map(|((g, p), t)| (format!("{g}/{p}"), t.clone()))
            .collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        pairs
    }

    /// Set a parameter by its `"graph/param"` key.  Returns true if found.
    pub fn set_param_by_key(&mut self, key: &str, tensor: Tensor<B>) -> bool {
        if let Some(pos) = key.find('/') {
            let graph = &key[..pos];
            let param = &key[pos + 1..];
            let k = (graph.to_string(), param.to_string());
            if let std::collections::hash_map::Entry::Occupied(mut e) = self.params.entry(k) {
                e.insert(tensor.set_variable());
                return true;
            }
        }
        false
    }

    /// The device this executor is running on.
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Execute a named graph with given inputs.
    pub fn run(
        &self,
        graph_name: &str,
        inputs: &HashMap<String, Tensor<B>>,
    ) -> Result<ExecResult<B>> {
        let graph = self.program.get_graph(graph_name).ok_or_else(|| {
            shrew_core::Error::msg(format!("Graph '{}' not found in program", graph_name))
        })?;
        self.execute_graph(graph, inputs)
    }

    /// Execute a graph, returning output tensors and all intermediate values.
    fn execute_graph(
        &self,
        graph: &IrGraph,
        inputs: &HashMap<String, Tensor<B>>,
    ) -> Result<ExecResult<B>> {
        let order = graph.topo_order();
        let mut values: HashMap<usize, Tensor<B>> = HashMap::new();

        // Map input nodes to their provided tensors
        for &input_id in &graph.inputs {
            let node = graph.node(input_id);
            if let Some(tensor) = inputs.get(&node.name) {
                values.insert(input_id.0, tensor.clone());
            }
        }

        // Map parameter nodes to their initialized tensors
        for param in &graph.params {
            let key = (graph.name.clone(), param.name.clone());
            if let Some(tensor) = self.params.get(&key) {
                values.insert(param.node_id.0, tensor.clone());
            }
        }

        // Execute each node in topological order
        for &node_id in &order {
            if values.contains_key(&node_id.0) {
                continue; // Already initialized (input or param)
            }
            let node = graph.node(node_id);
            let result = self.execute_node(graph, node, &values)?;
            values.insert(node_id.0, result);
        }

        // Collect outputs
        let mut outputs = HashMap::new();
        for output in &graph.outputs {
            if let Some(tensor) = values.get(&output.node_id.0) {
                outputs.insert(output.name.clone(), tensor.clone());
            }
        }

        Ok(ExecResult { outputs, values })
    }

    /// Execute a single node given its inputs' current values.
    fn execute_node(
        &self,
        _graph: &IrGraph,
        node: &IrNode,
        values: &HashMap<usize, Tensor<B>>,
    ) -> Result<Tensor<B>> {
        // Collect input tensors for this node
        let input_tensors: Vec<&Tensor<B>> = node
            .inputs
            .iter()
            .filter_map(|id| values.get(&id.0))
            .collect();

        match &node.op {
            // ── Identity: pass-through ──
            OpKind::Identity => input_tensors.first().map(|t| (*t).clone()).ok_or_else(|| {
                shrew_core::Error::msg(format!("Identity node '{}' has no input", node.name))
            }),

            // ── Unary ops ──
            OpKind::Neg => unary(&input_tensors, &node.name, |t| t.neg()),
            OpKind::Relu => unary(&input_tensors, &node.name, |t| t.relu()),
            OpKind::Gelu => unary(&input_tensors, &node.name, |t| t.gelu()),
            OpKind::Silu => unary(&input_tensors, &node.name, |t| t.silu()),
            OpKind::Sigmoid => unary(&input_tensors, &node.name, |t| t.sigmoid()),
            OpKind::Tanh => unary(&input_tensors, &node.name, |t| t.tanh()),
            OpKind::Exp => unary(&input_tensors, &node.name, |t| t.exp()),
            OpKind::Log => unary(&input_tensors, &node.name, |t| t.log()),
            OpKind::Sqrt => unary(&input_tensors, &node.name, |t| t.sqrt()),

            // ── Transpose ──
            OpKind::Transpose => {
                let t = require_input(&input_tensors, 0, &node.name)?;
                let rank = t.rank();
                if rank < 2 {
                    return Err(shrew_core::Error::msg(format!(
                        "Transpose requires rank >= 2, got {} for '{}'",
                        rank, node.name
                    )));
                }
                t.transpose(rank - 2, rank - 1)
            }

            // ── Binary ops ──
            OpKind::Add => binary(&input_tensors, &node.name, |a, b| a.add(b)),
            OpKind::Sub => binary(&input_tensors, &node.name, |a, b| a.sub(b)),
            OpKind::Mul => binary(&input_tensors, &node.name, |a, b| a.mul(b)),
            OpKind::Div => binary(&input_tensors, &node.name, |a, b| a.div(b)),
            OpKind::MatMul => binary(&input_tensors, &node.name, |a, b| a.matmul(b)),

            // ── Pow: x^y via exp(y * ln(x)) ──
            OpKind::Pow => {
                let base = require_input(&input_tensors, 0, &node.name)?;
                let exp_t = require_input(&input_tensors, 1, &node.name)?;
                // x^y = exp(y * ln(x))
                base.log()?.mul(exp_t)?.exp()
            }

            // ── Mod: a - floor(a / b) * b ──
            OpKind::Mod => {
                let a = require_input(&input_tensors, 0, &node.name)?;
                let b = require_input(&input_tensors, 1, &node.name)?;
                let quotient = a.div(b)?.floor()?;
                let product = quotient.mul(b)?;
                a.sub(&product)
            }

            // ── Reduction ops ──
            OpKind::Sum { dims, keepdim } => {
                let t = require_input(&input_tensors, 0, &node.name)?;
                if dims.is_empty() || (dims.len() == 1 && dims[0] == -1) {
                    t.sum_all()
                } else {
                    let dim = resolve_neg_dim(dims[0], t.rank());
                    t.sum(dim, *keepdim)
                }
            }

            OpKind::Mean { dims, keepdim } => {
                let t = require_input(&input_tensors, 0, &node.name)?;
                if dims.is_empty() || (dims.len() == 1 && dims[0] == -1) {
                    t.mean_all()
                } else {
                    let dim = resolve_neg_dim(dims[0], t.rank());
                    t.mean(dim, *keepdim)
                }
            }

            OpKind::Max { dim, keepdim } => {
                let t = require_input(&input_tensors, 0, &node.name)?;
                let d = resolve_neg_dim(*dim, t.rank());
                t.max(d, *keepdim)
            }

            OpKind::Min { dim, keepdim } => {
                let t = require_input(&input_tensors, 0, &node.name)?;
                let d = resolve_neg_dim(*dim, t.rank());
                t.min(d, *keepdim)
            }

            OpKind::Variance { dims, keepdim } => {
                let t = require_input(&input_tensors, 0, &node.name)?;
                if dims.is_empty() {
                    t.var(0, *keepdim)
                } else {
                    let dim = resolve_neg_dim(dims[0], t.rank());
                    t.var(dim, *keepdim)
                }
            }

            // ── Softmax ──
            OpKind::Softmax { dim } => {
                let t = require_input(&input_tensors, 0, &node.name)?;
                let d = resolve_neg_dim(*dim, t.rank());
                t.softmax(d)
            }

            // ── Shape ops ──
            OpKind::Reshape { target_shape } | OpKind::View { target_shape } => {
                let t = require_input(&input_tensors, 0, &node.name)?;
                let shape = self.resolve_shape_vec(target_shape)?;
                t.reshape(shape)
            }

            OpKind::Permute { dims: perm_dims } => {
                let t = require_input(&input_tensors, 0, &node.name)?;
                // Apply successive transpositions to achieve the permutation
                let mut result = t.clone();
                let mut current: Vec<usize> = (0..t.rank()).collect();
                for i in 0..perm_dims.len() {
                    let target = perm_dims[i] as usize;
                    if current[i] != target {
                        let j = current.iter().position(|&x| x == target).ok_or_else(|| {
                            shrew_core::Error::msg(format!(
                                "permute: dimension {} not found in current layout",
                                target
                            ))
                        })?;
                        result = result.transpose(i, j)?;
                        current.swap(i, j);
                    }
                }
                Ok(result)
            }

            OpKind::Expand { target_shape } => {
                let t = require_input(&input_tensors, 0, &node.name)?;
                let shape = self.resolve_shape_vec(target_shape)?;
                t.expand(shape)
            }

            OpKind::Concat { dim } => {
                if input_tensors.is_empty() {
                    return Err(shrew_core::Error::msg(format!(
                        "Concat node '{}' has no inputs",
                        node.name
                    )));
                }
                let owned: Vec<Tensor<B>> = input_tensors.iter().map(|t| (*t).clone()).collect();
                Tensor::<B>::cat(&owned, *dim as usize)
            }

            // ── Embedding ──
            // Convention: embedding(indices, weight_table)
            OpKind::Embedding => {
                let indices = require_input(&input_tensors, 0, &node.name)?;
                let table = require_input(&input_tensors, 1, &node.name)?;
                let emb = Embedding::<B>::from_tensor(table.clone())?;
                emb.forward(indices)
            }

            // ── Linear ──
            // Convention: linear(input, weight) or linear(input, weight, bias)
            OpKind::Linear { bias } => {
                let input = require_input(&input_tensors, 0, &node.name)?;
                let weight = require_input(&input_tensors, 1, &node.name)?;
                if *bias && input_tensors.len() >= 3 {
                    let bias_t = require_input(&input_tensors, 2, &node.name)?;
                    let lin = Linear::<B>::from_tensors(weight.clone(), Some(bias_t.clone()))?;
                    lin.forward(input)
                } else {
                    let lin = Linear::<B>::from_tensors(weight.clone(), None)?;
                    lin.forward(input)
                }
            }

            // ── LayerNorm ──
            // Convention: layer_norm(input, weight, bias)
            OpKind::LayerNorm { eps } => {
                let input = require_input(&input_tensors, 0, &node.name)?;
                let weight = require_input(&input_tensors, 1, &node.name)?;
                let bias_t = require_input(&input_tensors, 2, &node.name)?;
                let ln = LayerNorm::<B>::from_tensors(weight.clone(), bias_t.clone(), *eps)?;
                ln.forward(input)
            }

            // ── MultiHeadAttention ──
            OpKind::MultiHeadAttention { n_heads } => {
                let input = require_input(&input_tensors, 0, &node.name)?;
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

            // ── TransformerBlock ──
            OpKind::TransformerBlock { n_heads } => {
                let input = require_input(&input_tensors, 0, &node.name)?;
                let dims = input.dims();
                if dims.len() != 3 {
                    return Err(shrew_core::Error::msg(format!(
                        "TransformerBlock expects [batch, seq, d_model], got {:?}",
                        dims
                    )));
                }
                let d_model = dims[2];
                let d_ff = d_model * 4;
                let block = TransformerBlock::<B>::new(
                    d_model,
                    *n_heads as usize,
                    d_ff,
                    true, // causal by default
                    input.dtype(),
                    input.device(),
                )?;
                block.forward(input)
            }

            // ── Dropout ──
            OpKind::Dropout { p } => {
                let input = require_input(&input_tensors, 0, &node.name)?;
                let dropout = Dropout::new(*p);
                if self.config.training {
                    dropout.forward_t(input)
                } else {
                    Ok(input.clone())
                }
            }

            // ── Loss functions ──
            OpKind::CrossEntropy => {
                let predictions = require_input(&input_tensors, 0, &node.name)?;
                let targets = require_input(&input_tensors, 1, &node.name)?;
                cross_entropy_loss(predictions, targets)
            }

            OpKind::MseLoss => {
                let predictions = require_input(&input_tensors, 0, &node.name)?;
                let targets = require_input(&input_tensors, 1, &node.name)?;
                mse_loss(predictions, targets)
            }

            // ── Comparison ops ──
            OpKind::Equal
            | OpKind::NotEqual
            | OpKind::Less
            | OpKind::Greater
            | OpKind::LessEqual
            | OpKind::GreaterEqual => {
                let lhs = require_input(&input_tensors, 0, &node.name)?;
                let rhs = require_input(&input_tensors, 1, &node.name)?;
                match &node.op {
                    OpKind::Equal => lhs.eq(rhs),
                    OpKind::NotEqual => lhs.ne(rhs),
                    OpKind::Less => lhs.lt(rhs),
                    OpKind::Greater => lhs.gt(rhs),
                    OpKind::LessEqual => lhs.le(rhs),
                    OpKind::GreaterEqual => lhs.ge(rhs),
                    _ => unreachable!(),
                }
            }

            // ── Constants ──
            OpKind::Constant(val) => self.materialize_constant(val, &node.output_type),

            // ── Repeat: execute body_op N times in sequence ──
            OpKind::Repeat { count, body_op } => {
                let input = require_input(&input_tensors, 0, &node.name)?;
                let mut current = input.clone();
                for _ in 0..*count {
                    current = self.execute_body_op(body_op, &current)?;
                }
                Ok(current)
            }

            // ── Call: execute another graph ──
            OpKind::Call { graph_name } => {
                // Build inputs for the sub-graph
                let sub_graph = self.program.get_graph(graph_name).ok_or_else(|| {
                    shrew_core::Error::msg(format!("Called graph '{}' not found", graph_name))
                })?;
                let mut sub_inputs = HashMap::new();
                for (i, &input_id) in sub_graph.inputs.iter().enumerate() {
                    let input_node = sub_graph.node(input_id);
                    if let Some(tensor) = input_tensors.get(i) {
                        sub_inputs.insert(input_node.name.clone(), (*tensor).clone());
                    }
                }
                let result = self.execute_graph(sub_graph, &sub_inputs)?;
                result.output().cloned().ok_or_else(|| {
                    shrew_core::Error::msg(format!(
                        "Called graph '{}' produced no output",
                        graph_name
                    ))
                })
            }

            // ── Range ──
            OpKind::Range => {
                // range(start, end) → 1D tensor [start, start+1, ..., end-1]
                let (start, end) = if input_tensors.len() >= 2 {
                    let s = input_tensors[0].to_scalar_f64()?;
                    let e = input_tensors[1].to_scalar_f64()?;
                    (s as i64, e as i64)
                } else if input_tensors.len() == 1 {
                    (0i64, input_tensors[0].to_scalar_f64()? as i64)
                } else {
                    // Try resolving from output type shape
                    match &node.output_type {
                        IrType::Tensor { shape, .. } => {
                            if let Some(Dim::Fixed(n)) = shape.first() {
                                (0, *n)
                            } else if let Some(Dim::Symbolic(name)) = shape.first() {
                                let n = self.resolve_symbolic(name)? as i64;
                                (0, n)
                            } else {
                                (0, 1)
                            }
                        }
                        _ => (0, 1),
                    }
                };
                let data: Vec<f64> = (start..end).map(|i| i as f64).collect();
                let len = data.len();
                Tensor::<B>::from_f64_slice(&data, len, CoreDType::I64, &self.device)
            }

            // ── BatchNorm ──
            // Convention: batch_norm(input, weight, bias)
            OpKind::BatchNorm { eps } => {
                let input = require_input(&input_tensors, 0, &node.name)?;
                if input_tensors.len() >= 3 {
                    let weight = require_input(&input_tensors, 1, &node.name)?;
                    let bias_t = require_input(&input_tensors, 2, &node.name)?;
                    let bn = shrew_nn::BatchNorm2d::<B>::from_tensors(
                        weight.clone(),
                        bias_t.clone(),
                        *eps,
                    )?;
                    bn.forward(input)
                } else {
                    // No weight/bias provided — create default BatchNorm from channels
                    let dims = input.dims();
                    if dims.len() != 4 {
                        return Err(shrew_core::Error::msg(format!(
                            "BatchNorm expects 4D input [N,C,H,W], got {:?}",
                            dims
                        )));
                    }
                    let c = dims[1];
                    let bn =
                        shrew_nn::BatchNorm2d::<B>::new(c, *eps, 0.1, input.dtype(), &self.device)?;
                    bn.forward(input)
                }
            }

            // ── Split ──
            OpKind::Split { dim, chunks } => {
                let input = require_input(&input_tensors, 0, &node.name)?;
                let d = resolve_neg_dim(*dim, input.rank());
                let result = input.chunk(*chunks as usize, d)?;
                // Return first chunk (Split in IR produces a single node)
                result
                    .into_iter()
                    .next()
                    .ok_or_else(|| shrew_core::Error::msg("Split produced no chunks"))
            }

            // ── Logical ops (on comparison results) ──
            OpKind::And => {
                let lhs = require_input(&input_tensors, 0, &node.name)?;
                let rhs = require_input(&input_tensors, 1, &node.name)?;
                // a AND b = (a != 0) & (b != 0) → element-wise min
                let a_data = lhs.to_f64_vec()?;
                let b_data = rhs.to_f64_vec()?;
                let result: Vec<f64> = a_data
                    .iter()
                    .zip(b_data.iter())
                    .map(|(&a, &b)| if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 })
                    .collect();
                let n = result.len();
                Tensor::<B>::from_f64_slice(&result, n, CoreDType::U8, &self.device)
            }
            OpKind::Or => {
                let lhs = require_input(&input_tensors, 0, &node.name)?;
                let rhs = require_input(&input_tensors, 1, &node.name)?;
                let a_data = lhs.to_f64_vec()?;
                let b_data = rhs.to_f64_vec()?;
                let result: Vec<f64> = a_data
                    .iter()
                    .zip(b_data.iter())
                    .map(|(&a, &b)| if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 })
                    .collect();
                let n = result.len();
                Tensor::<B>::from_f64_slice(&result, n, CoreDType::U8, &self.device)
            }
            OpKind::Not => {
                let input = require_input(&input_tensors, 0, &node.name)?;
                let data = input.to_f64_vec()?;
                let result: Vec<f64> = data
                    .iter()
                    .map(|&v| if v == 0.0 { 1.0 } else { 0.0 })
                    .collect();
                let n = result.len();
                Tensor::<B>::from_f64_slice(&result, n, CoreDType::U8, &self.device)
            }

            // ── Custom op ──
            OpKind::Custom { name, .. } => {
                match name.as_str() {
                    // Fused matmul + add: a.matmul(b) + c (no weight transpose)
                    "fused_matmul_add" => {
                        let a = require_input(&input_tensors, 0, &node.name)?;
                        let b = require_input(&input_tensors, 1, &node.name)?;
                        let c = require_input(&input_tensors, 2, &node.name)?;
                        a.matmul(b)?.add(c)
                    }
                    // Fused add + relu
                    "fused_add_relu" => {
                        let a = require_input(&input_tensors, 0, &node.name)?;
                        let b = require_input(&input_tensors, 1, &node.name)?;
                        a.add(b)?.relu()
                    }
                    // Fused sub + relu
                    "fused_sub_relu" => {
                        let a = require_input(&input_tensors, 0, &node.name)?;
                        let b = require_input(&input_tensors, 1, &node.name)?;
                        a.sub(b)?.relu()
                    }
                    // Fused matmul + relu
                    "fused_matmul_relu" => {
                        let a = require_input(&input_tensors, 0, &node.name)?;
                        let b = require_input(&input_tensors, 1, &node.name)?;
                        a.matmul(b)?.relu()
                    }
                    _ => Err(shrew_core::Error::msg(format!(
                        "Custom op '{}' is not implemented in the executor",
                        name
                    ))),
                }
            }
        }
    }

    /// Execute a body op (used inside Repeat).
    fn execute_body_op(&self, op: &OpKind, input: &Tensor<B>) -> Result<Tensor<B>> {
        match op {
            OpKind::TransformerBlock { n_heads } => {
                let dims = input.dims();
                if dims.len() != 3 {
                    return Err(shrew_core::Error::msg(format!(
                        "TransformerBlock expects [batch, seq, d_model], got {:?}",
                        dims
                    )));
                }
                let d_model = dims[2];
                let d_ff = d_model * 4;
                let block = TransformerBlock::<B>::new(
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
            // For other repeated ops, dispatch through the main execute_node
            // infrastructure by returning an error so the caller knows.
            _ => Err(shrew_core::Error::msg(format!(
                "Unsupported op in Repeat body: {:?}. \
                 Only TransformerBlock and MultiHeadAttention are supported.",
                op
            ))),
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Parameter initialization
    // ─────────────────────────────────────────────────────────────────────

    /// Initialize all parameters across all graphs.
    fn init_all_params(&mut self) -> Result<()> {
        let graphs: Vec<(String, Vec<_>)> = self
            .program
            .graphs
            .iter()
            .map(|g| {
                (
                    g.name.clone(),
                    g.params
                        .iter()
                        .map(|p| (p.name.clone(), p.ty.clone(), p.init.clone(), p.frozen))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();

        for (graph_name, params) in &graphs {
            for (param_name, ty, init, frozen) in params {
                let tensor = self.init_param(ty, init, *frozen)?;
                self.params
                    .insert((graph_name.clone(), param_name.clone()), tensor);
            }
        }
        Ok(())
    }

    /// Initialize a single parameter tensor based on its type and init strategy.
    fn init_param(&self, ty: &IrType, init: &InitStrategy, frozen: bool) -> Result<Tensor<B>> {
        let (shape, dtype) = self.resolve_type(ty)?;
        let tensor = match init {
            InitStrategy::Zeros => Tensor::<B>::zeros(shape, dtype, &self.device)?,
            InitStrategy::Ones => Tensor::<B>::ones(shape, dtype, &self.device)?,
            InitStrategy::Normal { mean, std } => {
                Tensor::<B>::randn(shape, dtype, &self.device)?.affine(*std, *mean)?
            }
            InitStrategy::Uniform { low, high } => {
                let range = high - low;
                Tensor::<B>::rand(shape, dtype, &self.device)?.affine(range, *low)?
            }
            InitStrategy::XavierUniform => {
                // Xavier uniform: U(-a, a) where a = sqrt(6 / (fan_in + fan_out))
                let (fan_in, fan_out) = compute_fans(&shape);
                let a = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
                Tensor::<B>::rand(shape, dtype, &self.device)?.affine(2.0 * a, -a)?
            }
            InitStrategy::XavierNormal => {
                // Xavier normal: N(0, std) where std = sqrt(2 / (fan_in + fan_out))
                let (fan_in, fan_out) = compute_fans(&shape);
                let std = (2.0_f64 / (fan_in + fan_out) as f64).sqrt();
                Tensor::<B>::randn(shape, dtype, &self.device)?.affine(std, 0.0)?
            }
            InitStrategy::KaimingUniform => {
                // Kaiming uniform: U(-bound, bound) where bound = sqrt(3 / fan_in)
                let (fan_in, _) = compute_fans(&shape);
                let bound = (3.0_f64 / fan_in as f64).sqrt();
                Tensor::<B>::rand(shape, dtype, &self.device)?.affine(2.0 * bound, -bound)?
            }
            InitStrategy::KaimingNormal => {
                // Kaiming normal: N(0, std) where std = sqrt(2 / fan_in)
                let (fan_in, _) = compute_fans(&shape);
                let std = (2.0_f64 / fan_in as f64).sqrt();
                Tensor::<B>::randn(shape, dtype, &self.device)?.affine(std, 0.0)?
            }
            InitStrategy::Custom(_) => Tensor::<B>::randn(shape, dtype, &self.device)?,
        };

        if frozen {
            Ok(tensor)
        } else {
            Ok(tensor.set_variable())
        }
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

    /// Collect parameters for a specific graph (for optimizer).
    pub fn graph_params(&self, graph_name: &str) -> Vec<Tensor<B>> {
        self.params
            .iter()
            .filter(|((g, _), _)| g == graph_name)
            .map(|(_, t)| t.clone())
            .collect()
    }

    // ─────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Resolve a Dim to a concrete usize.
    fn resolve_dim(&self, dim: &Dim) -> Result<usize> {
        match dim {
            Dim::Fixed(n) => Ok(*n as usize),
            Dim::Symbolic(name) => self.resolve_symbolic(name),
            Dim::Dynamic => Err(shrew_core::Error::msg(
                "Cannot resolve dynamic dimension at runtime",
            )),
        }
    }

    /// Resolve a symbolic dimension name.
    fn resolve_symbolic(&self, name: &str) -> Result<usize> {
        // Try runtime config
        if let Some(&val) = self.config.dims.get(name) {
            return Ok(val);
        }
        // Try program config
        if let Some(ConfigValue::Int(n)) = self.program.config.get(name) {
            return Ok(*n as usize);
        }
        Err(shrew_core::Error::msg(format!(
            "Unresolved symbolic dimension: '{}'. Set it via RuntimeConfig::set_dim()",
            name
        )))
    }

    /// Resolve an IrType to a concrete (Shape, CoreDType).
    fn resolve_type(&self, ty: &IrType) -> Result<(shrew_core::Shape, CoreDType)> {
        match ty {
            IrType::Tensor { shape, dtype } => {
                let dims: Vec<usize> = shape
                    .iter()
                    .map(|d| self.resolve_dim(d))
                    .collect::<Result<Vec<_>>>()?;
                let core_dtype = ir_dtype_to_core(*dtype)?;
                Ok((shrew_core::Shape::new(dims), core_dtype))
            }
            IrType::Scalar(dtype) => {
                let core_dtype = ir_dtype_to_core(*dtype)?;
                Ok((shrew_core::Shape::new(vec![1]), core_dtype))
            }
            IrType::Int => Ok((shrew_core::Shape::new(vec![1]), CoreDType::I64)),
            _ => Ok((shrew_core::Shape::new(vec![1]), self.config.default_dtype)),
        }
    }

    /// Resolve a Vec<Dim> to a concrete shape tuple.
    fn resolve_shape_vec(&self, dims: &[Dim]) -> Result<Vec<usize>> {
        dims.iter().map(|d| self.resolve_dim(d)).collect()
    }

    /// Materialize a constant value as a tensor.
    fn materialize_constant(&self, val: &ConstantValue, ty: &IrType) -> Result<Tensor<B>> {
        match val {
            ConstantValue::Int(n) => {
                Tensor::<B>::from_f64_slice(&[*n as f64], 1, CoreDType::I64, &self.device)
            }
            ConstantValue::Float(f) => Tensor::<B>::from_f64_slice(
                &[*f],
                1,
                ir_type_dtype(ty, self.config.default_dtype)?,
                &self.device,
            ),
            ConstantValue::Bool(b) => Tensor::<B>::from_f64_slice(
                &[if *b { 1.0 } else { 0.0 }],
                1,
                CoreDType::U8,
                &self.device,
            ),
            ConstantValue::Str(_) => {
                // Strings can't be tensors — return a dummy scalar
                Tensor::<B>::zeros(1, self.config.default_dtype, &self.device)
            }
            ConstantValue::Null => Tensor::<B>::zeros(1, self.config.default_dtype, &self.device),
        }
    }
}

// Free helpers

/// Convert IR DType to core DType.
pub fn ir_dtype_to_core(dt: IrDType) -> Result<CoreDType> {
    match dt {
        IrDType::F32 => Ok(CoreDType::F32),
        IrDType::F64 => Ok(CoreDType::F64),
        IrDType::U8 => Ok(CoreDType::U8),
        IrDType::U32 => Ok(CoreDType::U32),
        IrDType::I64 => Ok(CoreDType::I64),
        // Map unsupported types to closest supported
        IrDType::F16 | IrDType::Bf16 => Ok(CoreDType::F32),
        IrDType::I8 | IrDType::I16 | IrDType::I32 => Ok(CoreDType::I64),
        IrDType::U16 => Ok(CoreDType::U32),
        IrDType::U64 => Ok(CoreDType::U32),
        IrDType::Bool => Ok(CoreDType::U8),
        _ => Err(shrew_core::Error::msg(format!(
            "Unsupported IR dtype: {dt}"
        ))),
    }
}

/// Extract dtype from IrType, with a fallback default.
fn ir_type_dtype(ty: &IrType, default: CoreDType) -> Result<CoreDType> {
    match ty {
        IrType::Tensor { dtype, .. } => ir_dtype_to_core(*dtype),
        IrType::Scalar(dtype) => ir_dtype_to_core(*dtype),
        _ => Ok(default),
    }
}

/// Resolve a negative dimension index.
fn resolve_neg_dim(dim: i64, rank: usize) -> usize {
    if dim < 0 {
        (rank as i64 + dim) as usize
    } else {
        dim as usize
    }
}

/// Require an input at a given index.
fn require_input<'a, B: Backend>(
    inputs: &[&'a Tensor<B>],
    idx: usize,
    node_name: &str,
) -> Result<&'a Tensor<B>> {
    inputs.get(idx).copied().ok_or_else(|| {
        shrew_core::Error::msg(format!(
            "Node '{}' expected input at index {}, but only {} inputs available",
            node_name,
            idx,
            inputs.len()
        ))
    })
}

/// Execute a unary op.
fn unary<B: Backend>(
    inputs: &[&Tensor<B>],
    node_name: &str,
    f: impl FnOnce(&Tensor<B>) -> Result<Tensor<B>>,
) -> Result<Tensor<B>> {
    let t = require_input(inputs, 0, node_name)?;
    f(t)
}

/// Execute a binary op.
fn binary<B: Backend>(
    inputs: &[&Tensor<B>],
    node_name: &str,
    f: impl FnOnce(&Tensor<B>, &Tensor<B>) -> Result<Tensor<B>>,
) -> Result<Tensor<B>> {
    let a = require_input(inputs, 0, node_name)?;
    let b = require_input(inputs, 1, node_name)?;
    f(a, b)
}

/// Compute (fan_in, fan_out) from a parameter shape.
///
/// Follows PyTorch conventions:
/// - 1-D (bias): fan_in = fan_out = shape[0]
/// - 2-D (linear weight): fan_in = shape[1], fan_out = shape[0]
/// - 3-D+ (conv weight): fan_in = shape[1] * receptive, fan_out = shape[0] * receptive
fn compute_fans(shape: &shrew_core::Shape) -> (usize, usize) {
    let dims = shape.dims();
    match dims.len() {
        0 => (1, 1),
        1 => (dims[0], dims[0]),
        2 => (dims[1], dims[0]),
        _ => {
            // Conv: [out_channels, in_channels, *kernel_size]
            let receptive: usize = dims[2..].iter().product();
            let fan_in = dims[1] * receptive;
            let fan_out = dims[0] * receptive;
            (fan_in, fan_out)
        }
    }
}
