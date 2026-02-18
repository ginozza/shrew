// Integration tests for validation, shape inference, and optimization passes

use shrew_ir::graph::*;
use shrew_ir::optimize::*;
use shrew_ir::shapes::{infer_graph_shapes, infer_shapes};
use shrew_ir::validate::{validate, validate_graph_standalone, ValidationErrorKind};
use shrew_ir::{lower, parse};

// Validation tests

#[test]
fn test_validate_good_graph() {
    let src = r#"
        @graph Forward {
            input x: Tensor<[Batch, 256], f32>;
            node y { op: relu(x); };
            output y;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();
    assert!(validate(&ir).is_ok());
}

#[test]
fn test_validate_no_outputs() {
    let mut graph = IrGraph::new("Bad");
    graph.add_node("x", OpKind::Identity, vec![], IrType::Unknown);
    // No outputs set

    let errors = validate_graph_standalone(&graph).unwrap_err();
    assert!(errors
        .iter()
        .any(|e| matches!(e.kind, ValidationErrorKind::NoOutputs)));
}

#[test]
fn test_validate_dangling_input() {
    let mut graph = IrGraph::new("Bad");
    let _x = graph.add_node("x", OpKind::Identity, vec![], IrType::Unknown);
    // Node y references NodeId(99) which doesn't exist
    let y = graph.add_node("y", OpKind::Relu, vec![NodeId(99)], IrType::Unknown);
    graph.add_output(y);

    let errors = validate_graph_standalone(&graph).unwrap_err();
    assert!(errors
        .iter()
        .any(|e| matches!(e.kind, ValidationErrorKind::DanglingInput { .. })));
}

#[test]
fn test_validate_binary_op_arity() {
    let mut graph = IrGraph::new("Bad");
    let x = graph.add_node("x", OpKind::Identity, vec![], IrType::Unknown);
    // Add with only 1 input instead of 2
    let y = graph.add_node("y", OpKind::Add, vec![x], IrType::Unknown);
    graph.add_output(y);

    let errors = validate_graph_standalone(&graph).unwrap_err();
    assert!(errors.iter().any(|e| matches!(
        e.kind,
        ValidationErrorKind::BinaryOpArity {
            expected: 2,
            got: 1
        }
    )));
}

#[test]
fn test_validate_unary_op_arity() {
    let mut graph = IrGraph::new("Bad");
    let x = graph.add_node("x", OpKind::Identity, vec![], IrType::Unknown);
    let y = graph.add_node("y", OpKind::Identity, vec![], IrType::Unknown);
    // Relu with 2 inputs instead of 1
    let z = graph.add_node("z", OpKind::Relu, vec![x, y], IrType::Unknown);
    graph.add_output(z);

    let errors = validate_graph_standalone(&graph).unwrap_err();
    assert!(errors
        .iter()
        .any(|e| matches!(e.kind, ValidationErrorKind::UnaryOpArity { .. })));
}

#[test]
fn test_validate_type_mismatch() {
    let mut graph = IrGraph::new("Bad");
    let x = graph.add_node(
        "x",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(10)],
            dtype: DType::F32,
        },
    );
    let y = graph.add_node(
        "y",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(10)],
            dtype: DType::F64, // Different dtype!
        },
    );
    let z = graph.add_node("z", OpKind::Add, vec![x, y], IrType::Unknown);
    graph.add_output(z);

    let errors = validate_graph_standalone(&graph).unwrap_err();
    assert!(errors
        .iter()
        .any(|e| matches!(e.kind, ValidationErrorKind::TypeMismatch { .. })));
}

#[test]
fn test_validate_training_graph_ref() {
    let src = r#"
        @graph Forward {
            input x: Tensor<[Batch, 256], f32>;
            output x;
        }
        @training {
            model: NonExistent;
            loss: mse;
            optimizer: { type: "SGD"; lr: 0.01; }
            epochs: 10;
            batch_size: 32;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();
    let errors = validate(&ir).unwrap_err();
    assert!(errors
        .iter()
        .any(|e| matches!(e.kind, ValidationErrorKind::TrainingGraphNotFound { .. })));
}

#[test]
fn test_validate_full_program_ok() {
    let src = r#"
        @graph Forward {
            input x: Tensor<[Batch, 256], f32>;
            node y { op: relu(x); };
            output y;
        }
        @training {
            model: Forward;
            loss: mse;
            optimizer: { type: "SGD"; lr: 0.01; }
            epochs: 10;
            batch_size: 32;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();
    assert!(validate(&ir).is_ok());
}

#[test]
fn test_validate_invalid_param_type() {
    let mut graph = IrGraph::new("Bad");
    let x = graph.add_node("x", OpKind::Identity, vec![], IrType::Int);
    graph.params.push(IrParam {
        node_id: x,
        name: "x".into(),
        ty: IrType::Int, // Should be Tensor
        init: InitStrategy::Zeros,
        frozen: false,
    });
    graph.add_output(x);

    let errors = validate_graph_standalone(&graph).unwrap_err();
    assert!(errors
        .iter()
        .any(|e| matches!(e.kind, ValidationErrorKind::ParamNotTensor { .. })));
}

// Shape inference tests

#[test]
fn test_shape_identity_propagation() {
    let mut graph = IrGraph::new("Test");
    let x = graph.add_node(
        "x",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Symbolic("Batch".into()), Dim::Fixed(256)],
            dtype: DType::F32,
        },
    );
    let y = graph.add_node("y", OpKind::Identity, vec![x], IrType::Unknown);
    graph.add_output(y);

    infer_graph_shapes(&mut graph);

    match &graph.node(y).output_type {
        IrType::Tensor { shape, dtype } => {
            assert_eq!(shape.len(), 2);
            assert!(matches!(&shape[0], Dim::Symbolic(s) if s == "Batch"));
            assert!(matches!(&shape[1], Dim::Fixed(256)));
            assert_eq!(*dtype, DType::F32);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn test_shape_relu_preserves() {
    let mut graph = IrGraph::new("Test");
    let x = graph.add_node(
        "x",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(32), Dim::Fixed(64)],
            dtype: DType::F32,
        },
    );
    let y = graph.add_node("y", OpKind::Relu, vec![x], IrType::Unknown);
    graph.add_output(y);

    infer_graph_shapes(&mut graph);

    match &graph.node(y).output_type {
        IrType::Tensor { shape, dtype } => {
            assert_eq!(shape, &vec![Dim::Fixed(32), Dim::Fixed(64)]);
            assert_eq!(*dtype, DType::F32);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn test_shape_add_broadcast() {
    let mut graph = IrGraph::new("Test");
    let x = graph.add_node(
        "x",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(32), Dim::Fixed(1)],
            dtype: DType::F32,
        },
    );
    let y = graph.add_node(
        "y",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(1), Dim::Fixed(64)],
            dtype: DType::F32,
        },
    );
    let z = graph.add_node("z", OpKind::Add, vec![x, y], IrType::Unknown);
    graph.add_output(z);

    infer_graph_shapes(&mut graph);

    match &graph.node(z).output_type {
        IrType::Tensor { shape, .. } => {
            assert_eq!(shape, &vec![Dim::Fixed(32), Dim::Fixed(64)]);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn test_shape_matmul() {
    let mut graph = IrGraph::new("Test");
    let a = graph.add_node(
        "a",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Symbolic("B".into()), Dim::Fixed(128), Dim::Fixed(64)],
            dtype: DType::F32,
        },
    );
    let b = graph.add_node(
        "b",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(64), Dim::Fixed(32)],
            dtype: DType::F32,
        },
    );
    let c = graph.add_node("c", OpKind::MatMul, vec![a, b], IrType::Unknown);
    graph.add_output(c);

    infer_graph_shapes(&mut graph);

    match &graph.node(c).output_type {
        IrType::Tensor { shape, dtype } => {
            assert_eq!(shape.len(), 3);
            assert!(matches!(&shape[0], Dim::Symbolic(s) if s == "B"));
            assert_eq!(shape[1], Dim::Fixed(128));
            assert_eq!(shape[2], Dim::Fixed(32));
            assert_eq!(*dtype, DType::F32);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn test_shape_transpose() {
    let mut graph = IrGraph::new("Test");
    let x = graph.add_node(
        "x",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(8), Dim::Fixed(16), Dim::Fixed(32)],
            dtype: DType::F32,
        },
    );
    let y = graph.add_node("y", OpKind::Transpose, vec![x], IrType::Unknown);
    graph.add_output(y);

    infer_graph_shapes(&mut graph);

    match &graph.node(y).output_type {
        IrType::Tensor { shape, .. } => {
            assert_eq!(shape, &vec![Dim::Fixed(8), Dim::Fixed(32), Dim::Fixed(16)]);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn test_shape_reduction_mean() {
    let mut graph = IrGraph::new("Test");
    let x = graph.add_node(
        "x",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(32), Dim::Fixed(64), Dim::Fixed(128)],
            dtype: DType::F32,
        },
    );
    let y = graph.add_node(
        "y",
        OpKind::Mean {
            dims: vec![1],
            keepdim: false,
        },
        vec![x],
        IrType::Unknown,
    );
    graph.add_output(y);

    infer_graph_shapes(&mut graph);

    match &graph.node(y).output_type {
        IrType::Tensor { shape, .. } => {
            assert_eq!(shape, &vec![Dim::Fixed(32), Dim::Fixed(128)]);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn test_shape_reduction_keepdim() {
    let mut graph = IrGraph::new("Test");
    let x = graph.add_node(
        "x",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(32), Dim::Fixed(64)],
            dtype: DType::F64,
        },
    );
    let y = graph.add_node(
        "y",
        OpKind::Sum {
            dims: vec![-1],
            keepdim: true,
        },
        vec![x],
        IrType::Unknown,
    );
    graph.add_output(y);

    infer_graph_shapes(&mut graph);

    match &graph.node(y).output_type {
        IrType::Tensor { shape, dtype } => {
            assert_eq!(shape, &vec![Dim::Fixed(32), Dim::Fixed(1)]);
            assert_eq!(*dtype, DType::F64);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn test_shape_comparison_bool_output() {
    let mut graph = IrGraph::new("Test");
    let x = graph.add_node(
        "x",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(10)],
            dtype: DType::F32,
        },
    );
    let y = graph.add_node(
        "y",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(10)],
            dtype: DType::F32,
        },
    );
    let z = graph.add_node("z", OpKind::Greater, vec![x, y], IrType::Unknown);
    graph.add_output(z);

    infer_graph_shapes(&mut graph);

    match &graph.node(z).output_type {
        IrType::Tensor { shape, dtype } => {
            assert_eq!(shape, &vec![Dim::Fixed(10)]);
            assert_eq!(*dtype, DType::Bool);
        }
        other => panic!("expected bool Tensor, got {other:?}"),
    }
}

#[test]
fn test_shape_loss_scalar() {
    let mut graph = IrGraph::new("Test");
    let pred = graph.add_node(
        "pred",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(32), Dim::Fixed(10)],
            dtype: DType::F32,
        },
    );
    let target = graph.add_node(
        "target",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(32)],
            dtype: DType::I64,
        },
    );
    let loss = graph.add_node(
        "loss",
        OpKind::CrossEntropy,
        vec![pred, target],
        IrType::Unknown,
    );
    graph.add_output(loss);

    infer_graph_shapes(&mut graph);

    match &graph.node(loss).output_type {
        IrType::Scalar(DType::F32) => {}
        other => panic!("expected Scalar(F32), got {other:?}"),
    }
}

#[test]
fn test_shape_infer_on_lowered_program() {
    let src = r#"
        @graph Forward {
            input x: Tensor<[Batch, 256], f32>;
            input y: Tensor<[Batch, 256], f32>;
            node sum { op: x + y; };
            node activated { op: relu(sum); };
            output activated;
        }
    "#;
    let ast = parse(src).unwrap();
    let mut ir = lower(&ast).unwrap();
    infer_shapes(&mut ir);

    let graph = &ir.graphs[0];
    // relu(sum) should have same shape as x+y, which broadcasts to [Batch, 256]
    let activated = graph.get_node("activated").unwrap();
    match &activated.output_type {
        IrType::Tensor { shape, dtype } => {
            assert_eq!(shape.len(), 2);
            assert_eq!(*dtype, DType::F32);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

// Optimization tests

#[test]
fn test_dce_removes_unused_nodes() {
    let mut graph = IrGraph::new("Test");
    let x = graph.add_node("x", OpKind::Identity, vec![], IrType::Unknown);
    let _unused = graph.add_node("unused", OpKind::Relu, vec![x], IrType::Unknown);
    let y = graph.add_node("y", OpKind::Gelu, vec![x], IrType::Unknown);
    graph.inputs.push(x);
    graph.add_output(y);

    assert_eq!(graph.len(), 3);
    let removed = eliminate_dead_code(&mut graph);
    assert!(removed > 0);
    assert_eq!(graph.len(), 2); // x and y remain
    assert!(graph.get_node("unused").is_none());
    assert!(graph.get_node("x").is_some());
    assert!(graph.get_node("y").is_some());
}

#[test]
fn test_dce_keeps_params() {
    let mut graph = IrGraph::new("Test");
    let x = graph.add_node("x", OpKind::Identity, vec![], IrType::Unknown);
    let w = graph.add_node(
        "w",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(256), Dim::Fixed(128)],
            dtype: DType::F32,
        },
    );
    let y = graph.add_node("y", OpKind::MatMul, vec![x, w], IrType::Unknown);
    graph.inputs.push(x);
    graph.add_output(y);
    graph.params.push(IrParam {
        node_id: w,
        name: "w".into(),
        ty: IrType::Unknown,
        init: InitStrategy::Zeros,
        frozen: false,
    });

    let removed = eliminate_dead_code(&mut graph);
    assert_eq!(removed, 0);
    assert_eq!(graph.len(), 3);
}

#[test]
fn test_identity_elimination() {
    let mut graph = IrGraph::new("Test");
    let x = graph.add_node(
        "x",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(32)],
            dtype: DType::F32,
        },
    );
    graph.inputs.push(x);

    // Add a passthrough identity: pass = identity(x)
    let pass = graph.add_node("pass", OpKind::Identity, vec![x], IrType::Unknown);
    // y = relu(pass)  →  should become relu(x)
    let y = graph.add_node("y", OpKind::Relu, vec![pass], IrType::Unknown);
    graph.add_output(y);

    let removed = eliminate_identities(&mut graph);
    assert!(removed > 0);

    // After elimination, "y" should take input directly from "x"
    let y_node = graph.get_node("y").unwrap();
    let input_name = &graph.node(y_node.inputs[0]).name;
    assert_eq!(input_name, "x");
}

#[test]
fn test_constant_folding() {
    let mut graph = IrGraph::new("Test");
    let a = graph.add_node(
        "a",
        OpKind::Constant(ConstantValue::Int(10)),
        vec![],
        IrType::Int,
    );
    let b = graph.add_node(
        "b",
        OpKind::Constant(ConstantValue::Int(20)),
        vec![],
        IrType::Int,
    );
    let c = graph.add_node("c", OpKind::Add, vec![a, b], IrType::Unknown);
    graph.add_output(c);

    let folded = fold_constants(&mut graph);
    assert_eq!(folded, 1);

    let c_node = graph.get_node("c").unwrap();
    match &c_node.op {
        OpKind::Constant(ConstantValue::Int(30)) => {}
        other => panic!("expected Constant(30), got {other:?}"),
    }
    assert!(c_node.inputs.is_empty()); // inputs cleared after folding
}

#[test]
fn test_constant_folding_float() {
    let mut graph = IrGraph::new("Test");
    let a = graph.add_node(
        "a",
        OpKind::Constant(ConstantValue::Float(2.5)),
        vec![],
        IrType::Scalar(DType::F64),
    );
    let b = graph.add_node(
        "b",
        OpKind::Constant(ConstantValue::Float(4.0)),
        vec![],
        IrType::Scalar(DType::F64),
    );
    let c = graph.add_node("c", OpKind::Mul, vec![a, b], IrType::Unknown);
    graph.add_output(c);

    let folded = fold_constants(&mut graph);
    assert_eq!(folded, 1);

    let c_node = graph.get_node("c").unwrap();
    match &c_node.op {
        OpKind::Constant(ConstantValue::Float(v)) => {
            assert!((v - 10.0).abs() < 1e-10);
        }
        other => panic!("expected Constant(10.0), got {other:?}"),
    }
}

#[test]
fn test_cse_elimination() {
    let mut graph = IrGraph::new("Test");
    let x = graph.add_node(
        "x",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(32)],
            dtype: DType::F32,
        },
    );
    graph.inputs.push(x);

    // Two identical relu ops on the same input
    let r1 = graph.add_node("r1", OpKind::Relu, vec![x], IrType::Unknown);
    let r2 = graph.add_node("r2", OpKind::Relu, vec![x], IrType::Unknown);

    // Both used as inputs to add
    let out = graph.add_node("out", OpKind::Add, vec![r1, r2], IrType::Unknown);
    graph.add_output(out);

    let eliminated = eliminate_common_subexprs(&mut graph);
    assert!(eliminated > 0);

    // After CSE, out should use r1 for both inputs
    let out_node = graph.get_node("out").unwrap();
    assert_eq!(out_node.inputs[0], out_node.inputs[1]);
}

#[test]
fn test_cse_doesnt_merge_dropout() {
    let mut graph = IrGraph::new("Test");
    let x = graph.add_node("x", OpKind::Identity, vec![], IrType::Unknown);
    graph.inputs.push(x);

    // Two dropout ops — should NOT be merged (different random masks)
    let d1 = graph.add_node("d1", OpKind::Dropout { p: 0.1 }, vec![x], IrType::Unknown);
    let d2 = graph.add_node("d2", OpKind::Dropout { p: 0.1 }, vec![x], IrType::Unknown);
    let out = graph.add_node("out", OpKind::Add, vec![d1, d2], IrType::Unknown);
    graph.add_output(out);

    let eliminated = eliminate_common_subexprs(&mut graph);
    assert_eq!(eliminated, 0); // Dropout has side effects
}

#[test]
fn test_optimize_full_pipeline() {
    let mut graph = IrGraph::new("Test");
    let x = graph.add_node(
        "x",
        OpKind::Identity,
        vec![],
        IrType::Tensor {
            shape: vec![Dim::Fixed(32)],
            dtype: DType::F32,
        },
    );
    graph.inputs.push(x);

    // dead code
    let _unused = graph.add_node("unused", OpKind::Neg, vec![x], IrType::Unknown);

    // redundant identity
    let pass = graph.add_node("pass", OpKind::Identity, vec![x], IrType::Unknown);

    // constant folding candidate
    let c1 = graph.add_node(
        "c1",
        OpKind::Constant(ConstantValue::Int(3)),
        vec![],
        IrType::Int,
    );
    let c2 = graph.add_node(
        "c2",
        OpKind::Constant(ConstantValue::Int(7)),
        vec![],
        IrType::Int,
    );
    let sum = graph.add_node("sum", OpKind::Add, vec![c1, c2], IrType::Unknown);

    let y = graph.add_node("y", OpKind::Relu, vec![pass], IrType::Unknown);
    graph.add_output(y);
    graph.add_output(sum);

    let before = graph.len();
    let stats = optimize_graph_with_stats(&mut graph);
    let after = graph.len();

    assert!(after < before, "optimization should reduce node count");
    assert!(
        stats.identities_removed > 0 || stats.dead_code_removed > 0 || stats.constants_folded > 0
    );

    // Check constant was folded
    let sum_node = graph.get_node("sum").unwrap();
    match &sum_node.op {
        OpKind::Constant(ConstantValue::Int(10)) => {}
        other => panic!("expected folded constant 10, got {other:?}"),
    }
}

#[test]
fn test_optimize_lowered_graph() {
    let src = r#"
        @graph Forward {
            input x: Tensor<[Batch, 256], f32>;
            input y: Tensor<[Batch, 256], f32>;
            node sum { op: x + y; };
            node activated { op: relu(sum); };
            output activated;
        }
    "#;
    let ast = parse(src).unwrap();
    let mut ir = lower(&ast).unwrap();

    // Should validate OK
    assert!(validate(&ir).is_ok());

    // Run shape inference
    infer_shapes(&mut ir);

    // Run optimization
    let _total = optimize(&mut ir);

    // The graph should still validate after optimization
    assert!(validate(&ir).is_ok());

    // Still has correct output
    assert_eq!(ir.graphs[0].outputs.len(), 1);
}

#[test]
fn test_full_pipeline_tiny_gpt() {
    let src = include_str!("../../../examples/tiny_gpt.sw");
    let ast = parse(src).unwrap();
    let mut ir = lower(&ast).unwrap();

    // Validate
    if let Err(errors) = validate(&ir) {
        for e in &errors {
            eprintln!("  pre-opt error: {e}");
        }
        panic!("pre-opt validation failed with {} errors", errors.len());
    }

    // Shape inference
    infer_shapes(&mut ir);

    // Optimize
    let _total = optimize(&mut ir);

    // Still valid post-optimization
    if let Err(errors) = validate(&ir) {
        for e in &errors {
            eprintln!("  validation error: {e}");
        }
        panic!("validation failed with {} errors", errors.len());
    }

    let graph = &ir.graphs[0];
    assert_eq!(graph.name, "Forward");
    assert!(!graph.outputs.is_empty());
    assert!(!graph.params.is_empty());
}

#[test]
fn test_opt_stats_display() {
    let stats = OptStats {
        dead_code_removed: 3,
        identities_removed: 2,
        constants_folded: 1,
        cse_eliminated: 0,
        ops_fused: 0,
    };
    let s = format!("{stats}");
    assert!(s.contains("dce: 3"));
    assert!(s.contains("identity: 2"));
    assert!(s.contains("const_fold: 1"));
    assert!(s.contains("cse: 0"));
}
