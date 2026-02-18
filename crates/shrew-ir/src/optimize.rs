// Graph Optimization — Transformations that simplify and optimize the IR graph
//
// These passes run after validation and shape inference, transforming the graph
// to reduce computation and memory usage. Each pass is idempotent and composable.
//
// Implemented passes:
//   1. Dead Code Elimination (DCE) — remove nodes not reachable from outputs
//   2. Identity Elimination — remove pass-through identity nodes
//   3. Constant Folding — evaluate constant sub-expressions at compile time
//   4. Common Sub-expression Elimination (CSE) — share identical computations
//   5. Operator Fusion — merge compatible adjacent operations
//
// The optimize() function runs all passes in a fixed-point loop until
// no more transformations apply.

use crate::graph::*;
use std::collections::{HashMap, HashSet};

// Public API

/// Run all optimization passes on every graph in the program.
/// Returns the total number of transformations applied.
pub fn optimize(program: &mut IrProgram) -> usize {
    let mut total = 0;
    for graph in &mut program.graphs {
        total += optimize_graph(graph);
    }
    total
}

/// Run all optimization passes on a single graph.
/// Runs passes in a loop until convergence (fixed point).
pub fn optimize_graph(graph: &mut IrGraph) -> usize {
    let mut total = 0;
    loop {
        let mut changed = 0;
        changed += eliminate_dead_code(graph);
        changed += eliminate_identities(graph);
        changed += fold_constants(graph);
        changed += eliminate_common_subexprs(graph);
        changed += fuse_operators(graph);
        if changed == 0 {
            break;
        }
        total += changed;
    }
    total
}

// Pass 1: Dead Code Elimination
//
// Walk backward from output nodes; any node not reachable is dead.
// Dead nodes are replaced with no-op markers and then compacted.

/// Remove nodes not reachable from any output. Returns count of removed nodes.
pub fn eliminate_dead_code(graph: &mut IrGraph) -> usize {
    if graph.nodes.is_empty() {
        return 0;
    }

    // Find all reachable nodes by walking backwards from outputs
    let mut reachable = HashSet::new();
    let mut stack: Vec<NodeId> = graph.outputs.iter().map(|o| o.node_id).collect();

    // Also keep param nodes alive
    for param in &graph.params {
        stack.push(param.node_id);
    }

    while let Some(id) = stack.pop() {
        if !reachable.insert(id) {
            continue;
        }
        if id.0 < graph.nodes.len() {
            for &inp in &graph.nodes[id.0].inputs {
                stack.push(inp);
            }
        }
    }

    let total = graph.nodes.len();
    let dead_count = total - reachable.len();
    if dead_count == 0 {
        return 0;
    }

    // Build compaction map: old_id → new_id
    let mut keep: Vec<bool> = vec![false; total];
    for &id in &reachable {
        keep[id.0] = true;
    }

    let mut old_to_new: Vec<Option<NodeId>> = vec![None; total];
    let mut new_id = 0usize;
    for old_id in 0..total {
        if keep[old_id] {
            old_to_new[old_id] = Some(NodeId(new_id));
            new_id += 1;
        }
    }

    // Compact nodes
    let mut new_nodes = Vec::with_capacity(reachable.len());
    for (old_id, node) in graph.nodes.drain(..).enumerate() {
        if let Some(nid) = old_to_new[old_id] {
            let mut node = node;
            node.id = nid;
            node.inputs = node
                .inputs
                .iter()
                .filter_map(|&inp| old_to_new[inp.0])
                .collect();
            new_nodes.push(node);
        }
    }
    graph.nodes = new_nodes;

    // Remap inputs, outputs, params, name_to_id
    graph.inputs = graph
        .inputs
        .iter()
        .filter_map(|&id| old_to_new[id.0])
        .collect();
    graph.outputs.retain(|o| old_to_new[o.node_id.0].is_some());
    for out in &mut graph.outputs {
        if let Some(new) = old_to_new[out.node_id.0] {
            out.node_id = new;
        }
    }

    for param in &mut graph.params {
        if let Some(new) = old_to_new[param.node_id.0] {
            param.node_id = new;
        }
    }
    graph.params.retain(|p| old_to_new[p.node_id.0].is_some());

    // Rebuild name_to_id
    graph.name_to_id.clear();
    for node in &graph.nodes {
        graph.name_to_id.insert(node.name.clone(), node.id);
    }

    dead_count
}

// Pass 2: Identity Elimination
//
// Identity nodes that aren't graph inputs or params can be collapsed: redirect
// all consumers to use the identity's input directly.

/// Eliminate redundant identity nodes. Returns count of identities removed.
pub fn eliminate_identities(graph: &mut IrGraph) -> usize {
    // Find identity nodes that can be removed
    // Keep: input identities, param identities, identities without inputs
    let input_set: HashSet<NodeId> = graph.inputs.iter().copied().collect();
    let param_set: HashSet<NodeId> = graph.params.iter().map(|p| p.node_id).collect();

    // Build identity map: node_id → its single input
    let mut identity_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in &graph.nodes {
        if matches!(node.op, OpKind::Identity)
            && node.inputs.len() == 1
            && !input_set.contains(&node.id)
            && !param_set.contains(&node.id)
        {
            identity_map.insert(node.id, node.inputs[0]);
        }
    }

    if identity_map.is_empty() {
        return 0;
    }

    // Resolve transitive chains: a→b→c → a→c
    let mut resolved: HashMap<NodeId, NodeId> = HashMap::new();
    for &id in identity_map.keys() {
        let mut target = id;
        let mut visited = HashSet::new();
        while let Some(&next) = identity_map.get(&target) {
            if !visited.insert(target) {
                break; // cycle guard
            }
            target = next;
        }
        resolved.insert(id, target);
    }

    let count = resolved.len();

    // Rewrite all node inputs
    for node in &mut graph.nodes {
        for inp in &mut node.inputs {
            if let Some(&target) = resolved.get(inp) {
                *inp = target;
            }
        }
    }

    // Rewrite graph outputs
    for out in &mut graph.outputs {
        if let Some(&target) = resolved.get(&out.node_id) {
            out.node_id = target;
        }
    }

    // Now run DCE to actually remove the unused identity nodes
    eliminate_dead_code(graph);

    count
}

// Pass 3: Constant Folding
//
// If a binary op has two constant inputs, evaluate the result at compile time.

/// Fold constant expressions. Returns count of folded nodes.
pub fn fold_constants(graph: &mut IrGraph) -> usize {
    let mut folded = 0;

    for i in 0..graph.nodes.len() {
        let node = &graph.nodes[i];

        // Only fold binary ops with two constant inputs
        if node.inputs.len() != 2 {
            continue;
        }

        let left_const = get_constant(&graph.nodes[node.inputs[0].0]);
        let right_const = get_constant(&graph.nodes[node.inputs[1].0]);

        let result = match (&node.op, left_const, right_const) {
            (OpKind::Add, Some(ConstantValue::Int(a)), Some(ConstantValue::Int(b))) => {
                Some(ConstantValue::Int(a + b))
            }
            (OpKind::Add, Some(ConstantValue::Float(a)), Some(ConstantValue::Float(b))) => {
                Some(ConstantValue::Float(a + b))
            }
            (OpKind::Sub, Some(ConstantValue::Int(a)), Some(ConstantValue::Int(b))) => {
                Some(ConstantValue::Int(a - b))
            }
            (OpKind::Sub, Some(ConstantValue::Float(a)), Some(ConstantValue::Float(b))) => {
                Some(ConstantValue::Float(a - b))
            }
            (OpKind::Mul, Some(ConstantValue::Int(a)), Some(ConstantValue::Int(b))) => {
                Some(ConstantValue::Int(a * b))
            }
            (OpKind::Mul, Some(ConstantValue::Float(a)), Some(ConstantValue::Float(b))) => {
                Some(ConstantValue::Float(a * b))
            }
            (OpKind::Div, Some(ConstantValue::Int(a)), Some(ConstantValue::Int(b))) if b != 0 => {
                Some(ConstantValue::Int(a / b))
            }
            (OpKind::Div, Some(ConstantValue::Float(a)), Some(ConstantValue::Float(b)))
                if b != 0.0 =>
            {
                Some(ConstantValue::Float(a / b))
            }
            _ => None,
        };

        if let Some(val) = result {
            graph.nodes[i].op = OpKind::Constant(val);
            graph.nodes[i].inputs.clear();
            folded += 1;
        }
    }

    if folded > 0 {
        eliminate_dead_code(graph);
    }

    folded
}

fn get_constant(node: &IrNode) -> Option<ConstantValue> {
    match &node.op {
        OpKind::Constant(v) => Some(v.clone()),
        _ => None,
    }
}

// Pass 4: Common Sub-expression Elimination (CSE)
//
// Two nodes with the same op and same inputs produce the same result.
// Keep the first, redirect consumers of the second.

/// Eliminate common sub-expressions. Returns count of CSE-eliminated nodes.
pub fn eliminate_common_subexprs(graph: &mut IrGraph) -> usize {
    // Collect input and param node IDs — these are semantically unique even
    // when they share the same OpSignature (e.g. Identity with no inputs).
    let protected: HashSet<NodeId> = graph
        .inputs
        .iter()
        .copied()
        .chain(graph.params.iter().map(|p| p.node_id))
        .collect();

    let mut canonical: HashMap<OpSignature, NodeId> = HashMap::new();
    let mut redirect: HashMap<NodeId, NodeId> = HashMap::new();

    for node in &graph.nodes {
        // Don't CSE nodes with side effects or that are inputs/params
        if has_side_effects(&node.op) || protected.contains(&node.id) {
            continue;
        }

        let sig = OpSignature {
            op: op_discriminant(&node.op),
            inputs: node.inputs.clone(),
        };

        if let Some(&existing_id) = canonical.get(&sig) {
            redirect.insert(node.id, existing_id);
        } else {
            canonical.insert(sig, node.id);
        }
    }

    if redirect.is_empty() {
        return 0;
    }

    let count = redirect.len();

    // Rewrite inputs
    for node in &mut graph.nodes {
        for inp in &mut node.inputs {
            if let Some(&target) = redirect.get(inp) {
                *inp = target;
            }
        }
    }

    // Rewrite outputs
    for out in &mut graph.outputs {
        if let Some(&target) = redirect.get(&out.node_id) {
            out.node_id = target;
        }
    }

    eliminate_dead_code(graph);

    count
}

/// A signature for CSE comparison.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OpSignature {
    op: String,
    inputs: Vec<NodeId>,
}

/// Get a deterministic string key for an op (including parameters).
fn op_discriminant(op: &OpKind) -> String {
    match op {
        OpKind::Add => "add".into(),
        OpKind::Sub => "sub".into(),
        OpKind::Mul => "mul".into(),
        OpKind::Div => "div".into(),
        OpKind::Mod => "mod".into(),
        OpKind::Pow => "pow".into(),
        OpKind::MatMul => "matmul".into(),
        OpKind::Neg => "neg".into(),
        OpKind::Relu => "relu".into(),
        OpKind::Gelu => "gelu".into(),
        OpKind::Silu => "silu".into(),
        OpKind::Sigmoid => "sigmoid".into(),
        OpKind::Tanh => "tanh".into(),
        OpKind::Exp => "exp".into(),
        OpKind::Log => "log".into(),
        OpKind::Sqrt => "sqrt".into(),
        OpKind::Transpose => "transpose".into(),
        OpKind::Not => "not".into(),
        OpKind::Identity => "identity".into(),
        OpKind::Softmax { dim } => format!("softmax_{dim}"),
        OpKind::LayerNorm { eps } => format!("layernorm_{eps}"),
        OpKind::BatchNorm { eps } => format!("batchnorm_{eps}"),
        OpKind::Sum { dims, keepdim } => format!("sum_{dims:?}_{keepdim}"),
        OpKind::Mean { dims, keepdim } => format!("mean_{dims:?}_{keepdim}"),
        OpKind::Max { dim, keepdim } => format!("max_{dim}_{keepdim}"),
        OpKind::Min { dim, keepdim } => format!("min_{dim}_{keepdim}"),
        OpKind::Variance { dims, keepdim } => format!("var_{dims:?}_{keepdim}"),
        OpKind::Dropout { p } => format!("dropout_{p}"),
        OpKind::Constant(v) => format!("const_{v}"),
        // Ops with side effects or custom behavior shouldn't CSE
        _ => format!("nocse_{op:?}"),
    }
}

/// Check if an op has side effects (shouldn't be CSE'd).
fn has_side_effects(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::Dropout { .. }  // Dropout is random
        | OpKind::Custom { .. } // Custom ops may have side effects
        | OpKind::Call { .. } // Graph calls may have side effects
    )
}

// Pass 5: Operator Fusion
//
// Fuse sequences of compatible operations into single fused ops:
//
//   MatMul + Add(bias)  → Linear { bias: true }
//   Add + Relu          → FusedAddRelu (custom op)
//   Sub + Relu          → FusedSubRelu
//   MatMul + Relu       → FusedMatMulRelu
//   Variance + Sqrt⁻¹ + Mul + Add  → LayerNorm (recognized pattern)
//
// Fusion reduces kernel launches and memory traffic on GPU backends.

/// Fuse compatible operator sequences. Returns count of fusions applied.
pub fn fuse_operators(graph: &mut IrGraph) -> usize {
    let mut fused = 0;
    fused += fuse_matmul_add(graph);
    fused += fuse_add_relu(graph);
    fused += fuse_matmul_relu(graph);
    fused
}

/// Fuse MatMul(x, w) → Add(bias) into fused_matmul_add custom op.
///
/// Note: we do NOT fuse into OpKind::Linear because Linear::forward()
/// transposes the weight (expects [out, in] layout), whereas raw matmul
/// in .sw uses [in, out]. Using a custom op preserves the original semantics
/// — a.matmul(b) + c with no implicit transpose.
fn fuse_matmul_add(graph: &mut IrGraph) -> usize {
    let mut fused = 0;
    let output_nodes: HashSet<NodeId> = graph.outputs.iter().map(|o| o.node_id).collect();

    // Find Add nodes whose first input comes from a MatMul
    let matmul_ids: HashSet<NodeId> = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, OpKind::MatMul))
        .map(|n| n.id)
        .collect();

    // Build consumer count: how many nodes consume each node
    let mut consumers: HashMap<NodeId, usize> = HashMap::new();
    for node in &graph.nodes {
        for &inp in &node.inputs {
            *consumers.entry(inp).or_insert(0) += 1;
        }
    }

    for i in 0..graph.nodes.len() {
        let node = &graph.nodes[i];
        if !matches!(node.op, OpKind::Add) || node.inputs.len() != 2 {
            continue;
        }

        let first_inp = node.inputs[0];
        let second_inp = node.inputs[1];

        // Pattern: Add(MatMul(x, w), bias) where MatMul has only 1 consumer
        if matmul_ids.contains(&first_inp)
            && consumers.get(&first_inp).copied().unwrap_or(0) == 1
            && !output_nodes.contains(&first_inp)
        {
            // Fuse: replace Add node -> fused_matmul_add custom op
            // Inputs become: [x, w, bias] from MatMul's [x, w] + Add's second input
            let matmul_inputs = graph.nodes[first_inp.0].inputs.clone();
            graph.nodes[i].op = OpKind::Custom {
                name: "fused_matmul_add".to_string(),
                attrs: HashMap::new(),
            };
            graph.nodes[i].inputs = vec![matmul_inputs[0], matmul_inputs[1], second_inp];
            graph.nodes[i].name = format!("{}_fused_matmul_add", graph.nodes[i].name);
            fused += 1;
        }
    }

    if fused > 0 {
        eliminate_dead_code(graph);
    }
    fused
}

/// Fuse Add/Sub + Relu into a single fused custom op.
fn fuse_add_relu(graph: &mut IrGraph) -> usize {
    let mut fused = 0;
    let output_nodes: HashSet<NodeId> = graph.outputs.iter().map(|o| o.node_id).collect();

    let add_sub_ids: HashSet<NodeId> = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, OpKind::Add | OpKind::Sub))
        .map(|n| n.id)
        .collect();

    let mut consumers: HashMap<NodeId, usize> = HashMap::new();
    for node in &graph.nodes {
        for &inp in &node.inputs {
            *consumers.entry(inp).or_insert(0) += 1;
        }
    }

    for i in 0..graph.nodes.len() {
        let node = &graph.nodes[i];
        if !matches!(node.op, OpKind::Relu) || node.inputs.len() != 1 {
            continue;
        }

        let inp = node.inputs[0];
        if add_sub_ids.contains(&inp)
            && consumers.get(&inp).copied().unwrap_or(0) == 1
            && !output_nodes.contains(&inp)
        {
            let is_add = matches!(graph.nodes[inp.0].op, OpKind::Add);
            let fused_name = if is_add {
                "fused_add_relu"
            } else {
                "fused_sub_relu"
            };
            let prev_inputs = graph.nodes[inp.0].inputs.clone();

            graph.nodes[i].op = OpKind::Custom {
                name: fused_name.to_string(),
                attrs: HashMap::new(),
            };
            graph.nodes[i].inputs = prev_inputs;
            graph.nodes[i].name = format!("{}_fused", graph.nodes[i].name);
            fused += 1;
        }
    }

    if fused > 0 {
        eliminate_dead_code(graph);
    }
    fused
}

/// Fuse MatMul + Relu into fused_matmul_relu.
fn fuse_matmul_relu(graph: &mut IrGraph) -> usize {
    let mut fused = 0;
    let output_nodes: HashSet<NodeId> = graph.outputs.iter().map(|o| o.node_id).collect();

    let matmul_ids: HashSet<NodeId> = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, OpKind::MatMul))
        .map(|n| n.id)
        .collect();

    let mut consumers: HashMap<NodeId, usize> = HashMap::new();
    for node in &graph.nodes {
        for &inp in &node.inputs {
            *consumers.entry(inp).or_insert(0) += 1;
        }
    }

    for i in 0..graph.nodes.len() {
        let node = &graph.nodes[i];
        if !matches!(node.op, OpKind::Relu) || node.inputs.len() != 1 {
            continue;
        }

        let inp = node.inputs[0];
        if matmul_ids.contains(&inp)
            && consumers.get(&inp).copied().unwrap_or(0) == 1
            && !output_nodes.contains(&inp)
        {
            let prev_inputs = graph.nodes[inp.0].inputs.clone();
            graph.nodes[i].op = OpKind::Custom {
                name: "fused_matmul_relu".to_string(),
                attrs: HashMap::new(),
            };
            graph.nodes[i].inputs = prev_inputs;
            graph.nodes[i].name = format!("{}_fused", graph.nodes[i].name);
            fused += 1;
        }
    }

    if fused > 0 {
        eliminate_dead_code(graph);
    }
    fused
}

// Convenience: Run specific passes

/// Statistics from optimization.
#[derive(Debug, Clone, Default)]
pub struct OptStats {
    pub dead_code_removed: usize,
    pub identities_removed: usize,
    pub constants_folded: usize,
    pub cse_eliminated: usize,
    pub ops_fused: usize,
}

impl std::fmt::Display for OptStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OptStats {{ dce: {}, identity: {}, const_fold: {}, cse: {}, fusion: {} }}",
            self.dead_code_removed,
            self.identities_removed,
            self.constants_folded,
            self.cse_eliminated,
            self.ops_fused,
        )
    }
}

/// Run all passes with detailed statistics.
pub fn optimize_graph_with_stats(graph: &mut IrGraph) -> OptStats {
    let mut stats = OptStats::default();
    loop {
        let dce = eliminate_dead_code(graph);
        let ident = eliminate_identities(graph);
        let cf = fold_constants(graph);
        let cse = eliminate_common_subexprs(graph);
        let fus = fuse_operators(graph);

        stats.dead_code_removed += dce;
        stats.identities_removed += ident;
        stats.constants_folded += cf;
        stats.cse_eliminated += cse;
        stats.ops_fused += fus;

        if dce + ident + cf + cse + fus == 0 {
            break;
        }
    }
    stats
}
