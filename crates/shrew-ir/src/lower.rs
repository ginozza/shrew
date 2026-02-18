// Lowering — AST → Graph IR
//
// This module transforms a parsed AST (shrew_ir::ast::Program) into a
// validated Graph IR (shrew_ir::graph::IrProgram). The lowering pass:
//
//   1. Resolves config values into a key-value map
//   2. Resolves type aliases
//   3. Lowers each @graph block into an IrGraph with typed nodes
//   4. Resolves operations from expressions (matmul, add, layer_norm, etc.)
//   5. Connects edges between nodes by name resolution
//   6. Lowers @training / @inference into config structs
//
// ERRORS: Lowering can fail if names are undefined, types mismatch, etc.
// We use the same Error type from error.rs.

use std::collections::HashMap;

use crate::ast::*;
use crate::error::{Error, ErrorKind, Result};
use crate::graph::*;
use crate::token::Span;

/// Lower a parsed AST program into a Graph IR program.
pub fn lower(program: &Program) -> Result<IrProgram> {
    let mut ctx = LowerCtx::new();
    ctx.lower_program(program)?;
    Ok(ctx.ir)
}

// Lowering context

struct LowerCtx {
    ir: IrProgram,
}

impl LowerCtx {
    fn new() -> Self {
        Self {
            ir: IrProgram::new(),
        }
    }

    fn lower_program(&mut self, program: &Program) -> Result<()> {
        // First pass: collect metadata, config, types (order-independent info)
        for item in &program.items {
            match item {
                TopLevel::Metadata(m) => self.lower_metadata(m)?,
                TopLevel::Config(c) => self.lower_config(c)?,
                TopLevel::Types(t) => self.lower_types(t)?,
                _ => {}
            }
        }

        // Second pass: lower graphs and other blocks
        for item in &program.items {
            match item {
                TopLevel::Graph(g) => {
                    let ir_graph = self.lower_graph(g)?;
                    self.ir.graphs.push(ir_graph);
                }
                TopLevel::Training(t) => {
                    self.ir.training = Some(self.lower_training(t)?);
                }
                TopLevel::Inference(i) => {
                    self.ir.inference = Some(self.lower_inference(i)?);
                }
                // Metadata, Config, Types already handled in first pass
                TopLevel::Metadata(_) | TopLevel::Config(_) | TopLevel::Types(_) => {}
                // Import, CustomOp, Metrics, Logging, Visualization — stored
                // but not deeply lowered yet (future work)
                _ => {}
            }
        }

        Ok(())
    }

    // @model

    fn lower_metadata(&mut self, meta: &MetadataBlock) -> Result<()> {
        for field in &meta.fields {
            let value = match &field.value {
                Literal::Str(s, _) => s.clone(),
                Literal::Int(n, _) => n.to_string(),
                Literal::Float(f, _) => f.to_string(),
                Literal::Bool(b, _) => b.to_string(),
                _ => format!("{:?}", field.value),
            };
            self.ir.metadata.insert(field.key.clone(), value);
        }
        Ok(())
    }
    
    // @config

    fn lower_config(&mut self, config: &ConfigBlock) -> Result<()> {
        for field in &config.fields {
            let value = self.eval_config_expr(&field.value)?;
            self.ir.config.insert(field.key.clone(), value);
        }
        Ok(())
    }

    /// Evaluate a config expression to a ConfigValue.
    /// Supports constant folding of arithmetic on integer/float literals.
    fn eval_config_expr(&self, expr: &Expr) -> Result<ConfigValue> {
        match expr {
            Expr::Int(n, _) => Ok(ConfigValue::Int(*n)),
            Expr::Float(f, _) => Ok(ConfigValue::Float(*f)),
            Expr::Str(s, _) => Ok(ConfigValue::Str(s.clone())),
            Expr::Bool(b, _) => Ok(ConfigValue::Bool(*b)),
            Expr::List(items, _) => {
                let vals: Result<Vec<_>> = items.iter().map(|e| self.eval_config_expr(e)).collect();
                Ok(ConfigValue::List(vals?))
            }
            Expr::Ident(name, _) => {
                // Reference to another config value
                if let Some(val) = self.ir.config.get(name) {
                    Ok(val.clone())
                } else {
                    // Treat as a string (e.g., symbolic dimension name)
                    Ok(ConfigValue::Str(name.clone()))
                }
            }
            Expr::Binary {
                left,
                op,
                right,
                span,
            } => {
                let l = self.eval_config_expr(left)?;
                let r = self.eval_config_expr(right)?;
                self.eval_binary_config(&l, *op, &r, *span)
            }
            Expr::Unary {
                op: UnaryOp::Neg,
                operand,
                ..
            } => {
                let val = self.eval_config_expr(operand)?;
                match val {
                    ConfigValue::Int(n) => Ok(ConfigValue::Int(-n)),
                    ConfigValue::Float(f) => Ok(ConfigValue::Float(-f)),
                    _ => Ok(val),
                }
            }
            _ => {
                // For complex expressions, store as string representation
                Ok(ConfigValue::Str(format!("{expr:?}")))
            }
        }
    }

    fn eval_binary_config(
        &self,
        left: &ConfigValue,
        op: BinOp,
        right: &ConfigValue,
        span: Span,
    ) -> Result<ConfigValue> {
        match (left, right) {
            (ConfigValue::Int(a), ConfigValue::Int(b)) => {
                let result = match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => {
                        if *b == 0 {
                            return Err(Error::new(
                                ErrorKind::Message("division by zero".into()),
                                span,
                            ));
                        }
                        a / b
                    }
                    BinOp::Mod => a % b,
                    BinOp::Pow => a.pow(*b as u32),
                    _ => return Ok(ConfigValue::Str(format!("{a} {op:?} {b}"))),
                };
                Ok(ConfigValue::Int(result))
            }
            (ConfigValue::Float(a), ConfigValue::Float(b)) => {
                let result = match op {
                    BinOp::Add => a + b,
                    BinOp::Sub => a - b,
                    BinOp::Mul => a * b,
                    BinOp::Div => a / b,
                    BinOp::Mod => a % b,
                    BinOp::Pow => a.powf(*b),
                    _ => return Ok(ConfigValue::Str(format!("{a} {op:?} {b}"))),
                };
                Ok(ConfigValue::Float(result))
            }
            (ConfigValue::Int(a), ConfigValue::Float(b)) => self.eval_binary_config(
                &ConfigValue::Float(*a as f64),
                op,
                &ConfigValue::Float(*b),
                span,
            ),
            (ConfigValue::Float(a), ConfigValue::Int(b)) => self.eval_binary_config(
                &ConfigValue::Float(*a),
                op,
                &ConfigValue::Float(*b as f64),
                span,
            ),
            _ => Ok(ConfigValue::Str(format!("{left:?} {op:?} {right:?}"))),
        }
    }

    // @types

    fn lower_types(&mut self, types: &TypesBlock) -> Result<()> {
        for def in &types.defs {
            let ty = self.lower_type_expr(&def.ty)?;
            self.ir.type_aliases.insert(def.name.clone(), ty);
        }
        Ok(())
    }

    fn lower_type_expr(&self, ty: &TypeExpr) -> Result<IrType> {
        match ty {
            TypeExpr::Tensor { dims, dtype, .. } => {
                let shape: Vec<Dim> = dims.iter().map(|d| self.lower_dim(d)).collect();
                let dt = lower_dtype(dtype);
                Ok(IrType::Tensor { shape, dtype: dt })
            }
            TypeExpr::Scalar(dt, _) => Ok(IrType::Scalar(lower_dtype(dt))),
            TypeExpr::Named(name, _) => {
                if let Some(resolved) = self.ir.type_aliases.get(name) {
                    Ok(resolved.clone())
                } else {
                    // Forward reference or unknown — keep as unknown
                    Ok(IrType::Unknown)
                }
            }
            TypeExpr::Dynamic(_) => Ok(IrType::Unknown),
            TypeExpr::IntDim(n, _) => {
                // A single integer dimension used as a type
                Ok(IrType::Tensor {
                    shape: vec![Dim::Fixed(*n)],
                    dtype: DType::F32,
                })
            }
            _ => Ok(IrType::Unknown),
        }
    }

    fn lower_dim(&self, dim: &Dimension) -> Dim {
        match dim {
            Dimension::Named(name, _) => {
                // Try to resolve symbolic dim from config
                if let Some(ConfigValue::Int(n)) = self.ir.config.get(name) {
                    Dim::Fixed(*n)
                } else {
                    Dim::Symbolic(name.clone())
                }
            }
            Dimension::Concrete(n, _) => Dim::Fixed(*n),
            Dimension::Dynamic(_) => Dim::Dynamic,
            Dimension::Inferred(_) => Dim::Dynamic,
            Dimension::Computed(expr, _) => {
                // Try to evaluate the expression as a constant integer dim
                if let Ok(ConfigValue::Int(n)) = self.eval_config_expr(expr) {
                    Dim::Fixed(n)
                } else if let Ok(ConfigValue::Float(f)) = self.eval_config_expr(expr) {
                    Dim::Fixed(f as i64)
                } else {
                    Dim::Dynamic
                }
            }
        }
    }

    // @graph

    fn lower_graph(&self, graph: &GraphBlock) -> Result<IrGraph> {
        let mut ir_graph = IrGraph::new(&graph.name);

        // Scope: name → NodeId for resolving references within this graph
        let mut scope: HashMap<String, NodeId> = HashMap::new();

        for stmt in &graph.body {
            match stmt {
                GraphStmt::Input(input) => {
                    let ty = self.lower_type_expr(&input.ty)?;
                    let id = ir_graph.add_node(&input.name, OpKind::Identity, vec![], ty);
                    ir_graph.inputs.push(id);
                    scope.insert(input.name.clone(), id);
                }
                GraphStmt::Param(param) => {
                    let ty = self.lower_type_expr(&param.ty)?;
                    let id = ir_graph.add_node(&param.name, OpKind::Identity, vec![], ty.clone());
                    scope.insert(param.name.clone(), id);

                    // Parse init strategy
                    let init = self.parse_init_strategy(&param.attrs);
                    let frozen = self.parse_frozen(&param.attrs);

                    ir_graph.params.push(IrParam {
                        node_id: id,
                        name: param.name.clone(),
                        ty,
                        init,
                        frozen,
                    });
                }
                GraphStmt::Node(node) => {
                    let (op, inputs) =
                        self.lower_node_body(&node.stmts, &mut ir_graph, &mut scope)?;
                    let output_type = node
                        .ty
                        .as_ref()
                        .map(|t| self.lower_type_expr(t))
                        .transpose()?
                        .unwrap_or(IrType::Unknown);

                    let id = ir_graph.add_node(&node.name, op, inputs, output_type);

                    // Transfer hints
                    for stmt in &node.stmts {
                        if let NodeStmt::Hint(hint, _) = stmt {
                            let ir_hint = match hint {
                                HintKind::RecomputeInBackward => IrHint::RecomputeInBackward,
                                HintKind::MustPreserve => IrHint::MustPreserve,
                                HintKind::InPlace => IrHint::InPlace,
                                HintKind::NoGrad => IrHint::NoGrad,
                                HintKind::Custom(s) => IrHint::Custom(s.clone()),
                            };
                            ir_graph.node_mut(id).hints.push(ir_hint);
                        }
                    }

                    // Transfer extra attributes
                    for stmt in &node.stmts {
                        if let NodeStmt::Attr(key, val, _) = stmt {
                            if let Some(attr) = self.expr_to_attr(val) {
                                ir_graph.node_mut(id).attrs.insert(key.clone(), attr);
                            }
                        }
                    }

                    scope.insert(node.name.clone(), id);
                }
                GraphStmt::Output(output) => {
                    // Derive a user-facing output name
                    let out_name = output.name.clone().unwrap_or_else(|| {
                        // Use the identifier name if it's a simple ident
                        if let Expr::Ident(ref ident, _) = output.expr {
                            ident.clone()
                        } else {
                            format!("__output_{}", ir_graph.outputs.len())
                        }
                    });

                    // Resolve the output expression to a node id
                    if let Some(id) = self.try_resolve_ident(&output.expr, &scope) {
                        ir_graph.outputs.push(IrOutput {
                            name: out_name,
                            node_id: id,
                        });
                    } else {
                        // Complex expression — lower it into a new node
                        let (op, inputs) =
                            self.lower_expr_to_op(&output.expr, &mut ir_graph, &mut scope)?;
                        let id = ir_graph.add_node(out_name.clone(), op, inputs, IrType::Unknown);
                        ir_graph.outputs.push(IrOutput {
                            name: out_name,
                            node_id: id,
                        });
                    }
                }
                GraphStmt::Assert(assert_stmt) => {
                    ir_graph.asserts.push(IrAssert {
                        message: assert_stmt.message.clone(),
                        expr_text: format!("{:?}", assert_stmt.condition),
                    });
                }
                GraphStmt::Check(_) => {
                    // Check blocks contain multiple asserts — lower each
                    // (Future: implement check block lowering)
                }
            }
        }

        Ok(ir_graph)
    }

    /// Parse param attributes to find the init strategy.
    fn parse_init_strategy(&self, attrs: &[ParamAttr]) -> InitStrategy {
        for attr in attrs {
            if attr.key == "init" {
                if let Expr::Str(s, _) = &attr.value {
                    return match s.as_str() {
                        "zeros" => InitStrategy::Zeros,
                        "ones" => InitStrategy::Ones,
                        "xavier_uniform" => InitStrategy::XavierUniform,
                        "xavier_normal" => InitStrategy::XavierNormal,
                        "kaiming_uniform" => InitStrategy::KaimingUniform,
                        "kaiming_normal" => InitStrategy::KaimingNormal,
                        s if s.starts_with("normal(") => {
                            // Parse "normal(mean, std)"
                            let inner = &s[7..s.len().saturating_sub(1)];
                            let parts: Vec<&str> = inner.split(',').collect();
                            if parts.len() == 2 {
                                let mean = parts[0].trim().parse().unwrap_or(0.0);
                                let std = parts[1].trim().parse().unwrap_or(0.02);
                                InitStrategy::Normal { mean, std }
                            } else {
                                InitStrategy::Custom(s.to_string())
                            }
                        }
                        s if s.starts_with("uniform(") => {
                            let inner = &s[8..s.len().saturating_sub(1)];
                            let parts: Vec<&str> = inner.split(',').collect();
                            if parts.len() == 2 {
                                let low = parts[0].trim().parse().unwrap_or(-1.0);
                                let high = parts[1].trim().parse().unwrap_or(1.0);
                                InitStrategy::Uniform { low, high }
                            } else {
                                InitStrategy::Custom(s.to_string())
                            }
                        }
                        other => InitStrategy::Custom(other.to_string()),
                    };
                }
            }
        }
        InitStrategy::Zeros // default
    }

    /// Parse param attributes to find frozen flag.
    fn parse_frozen(&self, attrs: &[ParamAttr]) -> bool {
        for attr in attrs {
            if attr.key == "frozen" {
                if let Expr::Bool(b, _) = &attr.value {
                    return *b;
                }
            }
        }
        false
    }

    /// Lower the body of a node { op: ...; input: ...; } into (OpKind, inputs).
    fn lower_node_body(
        &self,
        stmts: &[NodeStmt],
        graph: &mut IrGraph,
        scope: &mut HashMap<String, NodeId>,
    ) -> Result<(OpKind, Vec<NodeId>)> {
        for stmt in stmts {
            if let NodeStmt::Op(expr, _) = stmt {
                return self.lower_expr_to_op(expr, graph, scope);
            }
        }
        // No op found — identity
        Ok((OpKind::Identity, vec![]))
    }

    /// Lower an expression into an (OpKind, inputs) pair.
    /// May create intermediate nodes in the graph for nested expressions.
    fn lower_expr_to_op(
        &self,
        expr: &Expr,
        graph: &mut IrGraph,
        scope: &mut HashMap<String, NodeId>,
    ) -> Result<(OpKind, Vec<NodeId>)> {
        match expr {
            // Function call: matmul(x, W), layer_norm(h, w, b, eps: 1e-5), etc.
            Expr::Call { func, args, span } => self.lower_call(func, args, graph, scope, *span),
            // Binary expression: tok_emb + pos_emb
            Expr::Binary {
                left, op, right, ..
            } => {
                let left_id = self.lower_expr_to_node(left, graph, scope)?;
                let right_id = self.lower_expr_to_node(right, graph, scope)?;

                let op_kind = match op {
                    BinOp::Add => OpKind::Add,
                    BinOp::Sub => OpKind::Sub,
                    BinOp::Mul => OpKind::Mul,
                    BinOp::Div => OpKind::Div,
                    BinOp::Mod => OpKind::Mod,
                    BinOp::Pow => OpKind::Pow,
                    BinOp::Eq => OpKind::Equal,
                    BinOp::Ne => OpKind::NotEqual,
                    BinOp::Lt => OpKind::Less,
                    BinOp::Gt => OpKind::Greater,
                    BinOp::Le => OpKind::LessEqual,
                    BinOp::Ge => OpKind::GreaterEqual,
                    BinOp::And => OpKind::And,
                    BinOp::Or => OpKind::Or,
                    _ => OpKind::Custom {
                        name: format!("{op:?}"),
                        attrs: HashMap::new(),
                    },
                };

                Ok((op_kind, vec![left_id, right_id]))
            }
            // Unary expression: -x, !cond
            Expr::Unary { op, operand, .. } => {
                let operand_id = self.lower_expr_to_node(operand, graph, scope)?;
                let op_kind = match op {
                    UnaryOp::Neg => OpKind::Neg,
                    UnaryOp::Not => OpKind::Not,
                    UnaryOp::BitNot => OpKind::Custom {
                        name: "bitnot".into(),
                        attrs: HashMap::new(),
                    },
                };
                Ok((op_kind, vec![operand_id]))
            }
            // Identifier reference: just references another node
            Expr::Ident(name, _) => {
                if let Some(&id) = scope.get(name) {
                    Ok((OpKind::Identity, vec![id]))
                } else {
                    Ok((OpKind::Identity, vec![]))
                }
            }
            // Repeat expression: repeat(4) { transformer_block(h, ...) }
            Expr::RepeatExpr { count, body, .. } => {
                let n = match count.as_ref() {
                    Expr::Int(n, _) => *n,
                    _ => 1,
                };
                let (inner_op, inner_inputs) = self.lower_expr_to_op(body, graph, scope)?;
                // Wrap the inner op in a Repeat, preserving the body operation
                Ok((
                    OpKind::Repeat {
                        count: n,
                        body_op: Box::new(inner_op),
                    },
                    inner_inputs,
                ))
            }
            // Constants
            Expr::Int(n, _) => Ok((OpKind::Constant(ConstantValue::Int(*n)), vec![])),
            Expr::Float(f, _) => Ok((OpKind::Constant(ConstantValue::Float(*f)), vec![])),
            Expr::Str(s, _) => Ok((OpKind::Constant(ConstantValue::Str(s.clone())), vec![])),
            Expr::Bool(b, _) => Ok((OpKind::Constant(ConstantValue::Bool(*b)), vec![])),
            Expr::Null(_) => Ok((OpKind::Constant(ConstantValue::Null), vec![])),
            // Fallback
            _ => Ok((
                OpKind::Custom {
                    name: format!("{expr:?}"),
                    attrs: HashMap::new(),
                },
                vec![],
            )),
        }
    }

    /// Lower an expression to a single NodeId, creating intermediate nodes as needed.
    fn lower_expr_to_node(
        &self,
        expr: &Expr,
        graph: &mut IrGraph,
        scope: &mut HashMap<String, NodeId>,
    ) -> Result<NodeId> {
        // Fast path: simple identifier — look up in scope
        if let Expr::Ident(name, _) = expr {
            if let Some(&id) = scope.get(name) {
                return Ok(id);
            }
        }

        // Complex expression: lower to (op, inputs) and create an anonymous node
        let (op, inputs) = self.lower_expr_to_op(expr, graph, scope)?;
        let name = format!("__anon_{}", graph.len());
        let id = graph.add_node(name, op, inputs, IrType::Unknown);
        Ok(id)
    }

    /// Lower a function call to (OpKind, inputs).
    /// Positional args are lowered to nodes (creating intermediates for nested calls).
    fn lower_call(
        &self,
        func: &str,
        args: &[Arg],
        graph: &mut IrGraph,
        scope: &mut HashMap<String, NodeId>,
        _span: Span,
    ) -> Result<(OpKind, Vec<NodeId>)> {
        // Resolve positional args to node IDs, creating intermediate nodes as needed
        let input_ids: Vec<NodeId> = args
            .iter()
            .filter(|a| a.name.is_none())
            .map(|a| self.lower_expr_to_node(&a.value, graph, scope))
            .collect::<Result<Vec<_>>>()?;

        // Collect named args
        let named: HashMap<&str, &Expr> = args
            .iter()
            .filter_map(|a| a.name.as_deref().map(|name| (name, &a.value)))
            .collect();

        let op = match func {
            //  Core ops 
            "matmul" | "mm" => OpKind::MatMul,
            "add" => OpKind::Add,
            "sub" => OpKind::Sub,
            "mul" => OpKind::Mul,
            "div" => OpKind::Div,

            //  Activations 
            "relu" => OpKind::Relu,
            "gelu" => OpKind::Gelu,
            "silu" | "swish" => OpKind::Silu,
            "sigmoid" => OpKind::Sigmoid,
            "tanh" => OpKind::Tanh,
            "exp" => OpKind::Exp,
            "log" => OpKind::Log,
            "sqrt" => OpKind::Sqrt,

            //  Softmax 
            "softmax" => {
                let dim = self.get_named_int(&named, "dim").unwrap_or(-1);
                OpKind::Softmax { dim }
            }

            //  Embedding 
            "embedding" | "Embedding" => OpKind::Embedding,

            //  Linear 
            "linear" | "Linear" => {
                let bias = named
                    .get("bias")
                    .is_none_or(|e| matches!(e, Expr::Bool(true, _)));
                OpKind::Linear { bias }
            }

            //  Normalization 
            "layer_norm" | "LayerNorm" => {
                let eps = self.get_named_float(&named, "eps").unwrap_or(1e-5);
                OpKind::LayerNorm { eps }
            }
            "batch_norm" | "BatchNorm" => {
                let eps = self.get_named_float(&named, "eps").unwrap_or(1e-5);
                OpKind::BatchNorm { eps }
            }

            //  Attention 
            "multi_head_attention" | "MultiHeadAttention" => {
                let n_heads = self.get_named_int(&named, "n_heads").unwrap_or(1);
                OpKind::MultiHeadAttention { n_heads }
            }
            "transformer_block" | "TransformerBlock" => {
                let n_heads = self.get_named_int(&named, "n_heads").unwrap_or(1);
                OpKind::TransformerBlock { n_heads }
            }

            //  Reduction 
            "sum" => {
                let dim = self.get_named_int(&named, "dim").unwrap_or(-1);
                let keepdim = self.get_named_bool(&named, "keepdim").unwrap_or(false);
                OpKind::Sum {
                    dims: vec![dim],
                    keepdim,
                }
            }
            "mean" => {
                let dim = self.get_named_int(&named, "dim").unwrap_or(-1);
                let keepdim = self.get_named_bool(&named, "keepdim").unwrap_or(false);
                OpKind::Mean {
                    dims: vec![dim],
                    keepdim,
                }
            }

            //  Shape ops 
            "transpose" => OpKind::Transpose,
            "concat" | "cat" => {
                let dim = self.get_named_int(&named, "dim").unwrap_or(0);
                OpKind::Concat { dim }
            }

            //  Dropout 
            "dropout" | "Dropout" => {
                let p = self.get_named_float(&named, "p").unwrap_or(0.0);
                OpKind::Dropout { p }
            }

            //  Loss 
            "cross_entropy" | "cross_entropy_loss" => OpKind::CrossEntropy,
            "mse_loss" => OpKind::MseLoss,

            //  Range 
            "range" => OpKind::Range,

            //  Fallback: custom/unknown op 
            other => OpKind::Custom {
                name: other.to_string(),
                attrs: named
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.to_string(),
                            self.expr_to_attr(v)
                                .unwrap_or(AttrValue::Str(format!("{v:?}"))),
                        )
                    })
                    .collect(),
            },
        };

        Ok((op, input_ids))
    }

    /// Try to resolve a simple identifier to a node ID (no node creation).
    fn try_resolve_ident(&self, expr: &Expr, scope: &HashMap<String, NodeId>) -> Option<NodeId> {
        match expr {
            Expr::Ident(name, _) => scope.get(name).copied(),
            _ => None,
        }
    }

    /// Extract a named integer argument from a call.
    fn get_named_int(&self, named: &HashMap<&str, &Expr>, key: &str) -> Option<i64> {
        named.get(key).and_then(|e| match e {
            Expr::Int(n, _) => Some(*n),
            _ => None,
        })
    }

    /// Extract a named float argument from a call.
    fn get_named_float(&self, named: &HashMap<&str, &Expr>, key: &str) -> Option<f64> {
        named.get(key).and_then(|e| match e {
            Expr::Float(f, _) => Some(*f),
            Expr::Int(n, _) => Some(*n as f64),
            _ => None,
        })
    }

    /// Extract a named boolean argument from a call.
    fn get_named_bool(&self, named: &HashMap<&str, &Expr>, key: &str) -> Option<bool> {
        named.get(key).and_then(|e| match e {
            Expr::Bool(b, _) => Some(*b),
            _ => None,
        })
    }

    /// Convert an expression to an attribute value.
    fn expr_to_attr(&self, expr: &Expr) -> Option<AttrValue> {
        match expr {
            Expr::Int(n, _) => Some(AttrValue::Int(*n)),
            Expr::Float(f, _) => Some(AttrValue::Float(*f)),
            Expr::Str(s, _) => Some(AttrValue::Str(s.clone())),
            Expr::Bool(b, _) => Some(AttrValue::Bool(*b)),
            Expr::List(items, _) => {
                let vals: Vec<AttrValue> =
                    items.iter().filter_map(|e| self.expr_to_attr(e)).collect();
                Some(AttrValue::List(vals))
            }
            _ => None,
        }
    }

    // @training

    fn lower_training(&self, training: &TrainingBlock) -> Result<TrainingConfig> {
        let mut model_graph = String::new();
        let mut loss = String::new();
        let mut optimizer = OptimizerConfig {
            kind: "SGD".into(),
            lr: 0.01,
            extra: HashMap::new(),
        };
        let mut lr_schedule = None;
        let mut grad_clip = None;
        let mut precision = "fp32".to_string();
        let mut epochs: i64 = 1;
        let mut batch_size: i64 = 1;
        let mut accumulation_steps: i64 = 1;

        for field in &training.fields {
            match field {
                TrainingField::Model(name, _) => model_graph = name.clone(),
                TrainingField::Loss(name, _) => loss = name.clone(),
                TrainingField::Optimizer(fields, _) => {
                    optimizer = self.lower_optimizer_config(fields)?;
                }
                TrainingField::LrSchedule(fields, _) => {
                    lr_schedule = Some(self.lower_lr_schedule_config(fields)?);
                }
                TrainingField::GradClip(fields, _) => {
                    grad_clip = Some(self.lower_grad_clip_config(fields)?);
                }
                TrainingField::Generic(f) => match f.key.as_str() {
                    "precision" => {
                        if let Ok(ConfigValue::Str(s)) = self.eval_config_expr(&f.value) {
                            precision = s;
                        }
                    }
                    "epochs" => {
                        if let Ok(ConfigValue::Int(n)) = self.eval_config_expr(&f.value) {
                            epochs = n;
                        }
                    }
                    "batch_size" => {
                        if let Ok(ConfigValue::Int(n)) = self.eval_config_expr(&f.value) {
                            batch_size = n;
                        }
                    }
                    "accumulation_steps" => {
                        if let Ok(ConfigValue::Int(n)) = self.eval_config_expr(&f.value) {
                            accumulation_steps = n;
                        }
                    }
                    _ => {}
                },
            }
        }

        Ok(TrainingConfig {
            model_graph,
            loss,
            optimizer,
            lr_schedule,
            grad_clip,
            precision,
            epochs,
            batch_size,
            accumulation_steps,
        })
    }

    fn lower_optimizer_config(&self, fields: &[ExprField]) -> Result<OptimizerConfig> {
        let mut kind = "SGD".to_string();
        let mut lr = 0.01;
        let mut extra = HashMap::new();

        for f in fields {
            match f.key.as_str() {
                "type" => {
                    if let Ok(ConfigValue::Str(s)) = self.eval_config_expr(&f.value) {
                        kind = s;
                    }
                }
                "lr" | "learning_rate" => {
                    if let Ok(val) = self.eval_config_expr(&f.value) {
                        match &val {
                            ConfigValue::Float(v) => lr = *v,
                            ConfigValue::Int(n) => lr = *n as f64,
                            _ => {}
                        }
                    }
                }
                other => {
                    if let Ok(val) = self.eval_config_expr(&f.value) {
                        extra.insert(other.to_string(), val);
                    }
                }
            }
        }

        Ok(OptimizerConfig { kind, lr, extra })
    }

    fn lower_lr_schedule_config(&self, fields: &[ExprField]) -> Result<LrScheduleConfig> {
        let mut kind = "constant".to_string();
        let mut extra = HashMap::new();

        for f in fields {
            match f.key.as_str() {
                "type" => {
                    if let Ok(ConfigValue::Str(s)) = self.eval_config_expr(&f.value) {
                        kind = s;
                    }
                }
                other => {
                    if let Ok(val) = self.eval_config_expr(&f.value) {
                        extra.insert(other.to_string(), val);
                    }
                }
            }
        }

        Ok(LrScheduleConfig { kind, extra })
    }

    fn lower_grad_clip_config(&self, fields: &[ExprField]) -> Result<GradClipConfig> {
        let mut kind = "none".to_string();
        let mut extra = HashMap::new();

        for f in fields {
            match f.key.as_str() {
                "type" => {
                    if let Ok(ConfigValue::Str(s)) = self.eval_config_expr(&f.value) {
                        kind = s;
                    }
                }
                other => {
                    if let Ok(val) = self.eval_config_expr(&f.value) {
                        extra.insert(other.to_string(), val);
                    }
                }
            }
        }

        Ok(GradClipConfig { kind, extra })
    }

    // @inference

    fn lower_inference(&self, inference: &InferenceBlock) -> Result<InferenceConfig> {
        let mut model_graph = String::new();
        let mut quantization = None;
        let mut generation = None;

        for field in &inference.fields {
            match field {
                InferenceField::Model(name, _) => model_graph = name.clone(),
                InferenceField::Quantization(fields, _) => {
                    let mut map = HashMap::new();
                    for f in fields {
                        if let Ok(val) = self.eval_config_expr(&f.value) {
                            map.insert(f.key.clone(), val);
                        }
                    }
                    quantization = Some(map);
                }
                InferenceField::Generation(fields, _) => {
                    let mut map = HashMap::new();
                    for f in fields {
                        if let Ok(val) = self.eval_config_expr(&f.value) {
                            map.insert(f.key.clone(), val);
                        }
                    }
                    generation = Some(map);
                }
                _ => {}
            }
        }

        Ok(InferenceConfig {
            model_graph,
            quantization,
            generation,
        })
    }
}

// Helpers

/// Convert AST DTypeKind to IR DType.
fn lower_dtype(dt: &DTypeKind) -> DType {
    match dt {
        DTypeKind::F16 => DType::F16,
        DTypeKind::F32 => DType::F32,
        DTypeKind::F64 => DType::F64,
        DTypeKind::Bf16 => DType::Bf16,
        DTypeKind::I8 => DType::I8,
        DTypeKind::I16 => DType::I16,
        DTypeKind::I32 => DType::I32,
        DTypeKind::I64 => DType::I64,
        DTypeKind::U8 => DType::U8,
        DTypeKind::U16 => DType::U16,
        DTypeKind::U32 => DType::U32,
        DTypeKind::U64 => DType::U64,
        DTypeKind::Bool => DType::Bool,
        DTypeKind::Complex64 => DType::Complex64,
        DTypeKind::Complex128 => DType::Complex128,
    }
}
