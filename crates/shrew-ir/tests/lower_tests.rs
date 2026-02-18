// Integration tests for AST → Graph IR lowering

use shrew_ir::graph::*;
use shrew_ir::{lower, parse};

// Metadata / Config lowering

#[test]
fn test_lower_metadata() {
    let src = r#"
        @model {
            name: "TestModel";
            version: "1.0";
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();
    assert_eq!(ir.metadata.get("name").unwrap(), "TestModel");
    assert_eq!(ir.metadata.get("version").unwrap(), "1.0");
}

#[test]
fn test_lower_config_simple() {
    let src = r#"
        @config {
            d_model: 768;
            dropout: 0.1;
            name: "test";
            flag: true;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();

    match ir.config.get("d_model").unwrap() {
        ConfigValue::Int(768) => {}
        other => panic!("expected Int(768), got {other:?}"),
    }
    match ir.config.get("dropout").unwrap() {
        ConfigValue::Float(f) => assert!((f - 0.1).abs() < 1e-10),
        other => panic!("expected Float(0.1), got {other:?}"),
    }
    match ir.config.get("name").unwrap() {
        ConfigValue::Str(s) => assert_eq!(s, "test"),
        other => panic!("expected Str, got {other:?}"),
    }
    match ir.config.get("flag").unwrap() {
        ConfigValue::Bool(true) => {}
        other => panic!("expected Bool(true), got {other:?}"),
    }
}

#[test]
fn test_lower_config_constant_folding() {
    let src = r#"
        @config {
            d_ff: 768 * 4;
            hidden: 256 + 512;
            ratio: 10 / 2;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();

    match ir.config.get("d_ff").unwrap() {
        ConfigValue::Int(3072) => {}
        other => panic!("expected Int(3072), got {other:?}"),
    }
    match ir.config.get("hidden").unwrap() {
        ConfigValue::Int(768) => {}
        other => panic!("expected Int(768), got {other:?}"),
    }
    match ir.config.get("ratio").unwrap() {
        ConfigValue::Int(5) => {}
        other => panic!("expected Int(5), got {other:?}"),
    }
}

// Type lowering

#[test]
fn test_lower_types() {
    let src = r#"
        @types {
            type Hidden = Tensor<[Batch, 768], f32>;
            type Logits = Tensor<[Batch, 50257], f64>;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();

    assert_eq!(ir.type_aliases.len(), 2);

    match ir.type_aliases.get("Hidden").unwrap() {
        IrType::Tensor { shape, dtype } => {
            assert_eq!(shape.len(), 2);
            assert!(matches!(&shape[0], Dim::Symbolic(s) if s == "Batch"));
            assert!(matches!(&shape[1], Dim::Fixed(768)));
            assert_eq!(*dtype, DType::F32);
        }
        other => panic!("expected Tensor type, got {other:?}"),
    }

    match ir.type_aliases.get("Logits").unwrap() {
        IrType::Tensor { shape, dtype } => {
            assert_eq!(shape.len(), 2);
            assert_eq!(*dtype, DType::F64);
        }
        other => panic!("expected Tensor type, got {other:?}"),
    }
}

#[test]
fn test_lower_types_config_dim_resolution() {
    // Config values should be resolved in dimensions
    let src = r#"
        @config {
            d_model: 256;
        }

        @types {
            type Hidden = Tensor<[Batch, d_model], f32>;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();

    match ir.type_aliases.get("Hidden").unwrap() {
        IrType::Tensor { shape, .. } => {
            // d_model should resolve to Fixed(256) from config
            assert!(matches!(&shape[0], Dim::Symbolic(s) if s == "Batch"));
            assert!(matches!(&shape[1], Dim::Fixed(256)));
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

// Graph lowering

#[test]
fn test_lower_simple_graph() {
    let src = r#"
        @graph Forward {
            input x: Tensor<[Batch, 768], f32>;
            output x;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();

    assert_eq!(ir.graphs.len(), 1);
    let graph = &ir.graphs[0];
    assert_eq!(graph.name, "Forward");
    assert_eq!(graph.inputs.len(), 1);
    assert_eq!(graph.outputs.len(), 1);
}

#[test]
fn test_lower_graph_with_params() {
    let src = r#"
        @graph Linear {
            input x: Tensor<[Batch, 768], f32>;

            param W: Tensor<[768, 256], f32> {
                init: "normal(0, 0.02)";
                frozen: false;
            };

            param b: Tensor<[256], f32> {
                init: "zeros";
                frozen: false;
            };

            node h {
                op: matmul(x, W);
            };

            output h;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();

    let graph = &ir.graphs[0];
    assert_eq!(graph.params.len(), 2);

    // W param
    assert_eq!(graph.params[0].name, "W");
    assert!(!graph.params[0].frozen);
    match &graph.params[0].init {
        InitStrategy::Normal { mean, std } => {
            assert!((mean - 0.0).abs() < 1e-10);
            assert!((std - 0.02).abs() < 1e-10);
        }
        other => panic!("expected Normal init, got {other:?}"),
    }

    // b param
    assert_eq!(graph.params[1].name, "b");
    assert!(matches!(graph.params[1].init, InitStrategy::Zeros));

    // h node should be MatMul
    let h = graph.get_node("h").unwrap();
    assert!(matches!(h.op, OpKind::MatMul));
    assert_eq!(h.inputs.len(), 2);
}

#[test]
fn test_lower_graph_binary_ops() {
    let src = r#"
        @graph AddGraph {
            input a: Tensor<[Batch, 256], f32>;
            input b: Tensor<[Batch, 256], f32>;

            node c {
                op: a + b;
            };

            output c;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();

    let graph = &ir.graphs[0];
    let c = graph.get_node("c").unwrap();
    assert!(matches!(c.op, OpKind::Add));
    assert_eq!(c.inputs.len(), 2);
}

#[test]
fn test_lower_graph_with_hints() {
    let src = r#"
        @graph WithHints {
            input x: Tensor<[Batch, 256], f32>;

            node h {
                op: expensive_op(x);
                @hint recompute_in_backward;
                @hint no_grad;
            };

            output h;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();

    let graph = &ir.graphs[0];
    let h = graph.get_node("h").unwrap();
    assert_eq!(h.hints.len(), 2);
    assert!(h.hints.contains(&IrHint::RecomputeInBackward));
    assert!(h.hints.contains(&IrHint::NoGrad));
}

#[test]
fn test_lower_graph_with_assert() {
    let src = r#"
        @graph Checked {
            input x: Tensor<[Batch, 256], f32>;
            @assert Batch > 0, "batch must be positive";
            output x;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();

    let graph = &ir.graphs[0];
    assert_eq!(graph.asserts.len(), 1);
    assert_eq!(
        graph.asserts[0].message.as_deref(),
        Some("batch must be positive")
    );
}

#[test]
fn test_lower_graph_function_calls() {
    let src = r#"
        @graph Attention {
            input x: Tensor<[Batch, SeqLen, 256], f32>;

            param w: Tensor<[256], f32> {
                init: "ones";
            };

            param b: Tensor<[256], f32> {
                init: "zeros";
            };

            node normed {
                op: layer_norm(x, w, b, eps: 1e-5);
            };

            node activated {
                op: relu(normed);
            };

            node soft {
                op: softmax(activated, dim: -1);
            };

            output soft;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();

    let graph = &ir.graphs[0];

    let normed = graph.get_node("normed").unwrap();
    assert!(matches!(normed.op, OpKind::LayerNorm { eps } if (eps - 1e-5).abs() < 1e-10));

    let activated = graph.get_node("activated").unwrap();
    assert!(matches!(activated.op, OpKind::Relu));

    let soft = graph.get_node("soft").unwrap();
    assert!(matches!(soft.op, OpKind::Softmax { dim: -1 }));
}

#[test]
fn test_lower_graph_topo_order() {
    let src = r#"
        @graph Forward {
            input x: Tensor<[Batch, 768], f32>;
            input y: Tensor<[Batch, 768], f32>;

            node sum {
                op: x + y;
            };

            node out {
                op: relu(sum);
            };

            output out;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();

    let graph = &ir.graphs[0];
    let order = graph.topo_order();

    // All 4 nodes should appear in topo order
    assert_eq!(order.len(), 4);

    // x and y (inputs) should come before sum, sum before out
    let x_pos = order
        .iter()
        .position(|id| graph.node(*id).name == "x")
        .unwrap();
    let y_pos = order
        .iter()
        .position(|id| graph.node(*id).name == "y")
        .unwrap();
    let sum_pos = order
        .iter()
        .position(|id| graph.node(*id).name == "sum")
        .unwrap();
    let out_pos = order
        .iter()
        .position(|id| graph.node(*id).name == "out")
        .unwrap();

    assert!(x_pos < sum_pos);
    assert!(y_pos < sum_pos);
    assert!(sum_pos < out_pos);
}

// Training config lowering

#[test]
fn test_lower_training() {
    let src = r#"
        @training {
            model: Forward;
            loss: cross_entropy;
            optimizer: {
                type: "AdamW";
                lr: 3e-4;
                weight_decay: 0.1;
            }
            lr_schedule: {
                type: "cosine";
                warmup_steps: 500;
            }
            grad_clip: {
                type: "norm";
                max_norm: 1.0;
            }
            precision: "bf16";
            epochs: 20;
            batch_size: 64;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();

    let training = ir.training.as_ref().unwrap();
    assert_eq!(training.model_graph, "Forward");
    assert_eq!(training.loss, "cross_entropy");
    assert_eq!(training.optimizer.kind, "AdamW");
    assert!((training.optimizer.lr - 3e-4).abs() < 1e-10);
    assert_eq!(training.precision, "bf16");
    assert_eq!(training.epochs, 20);
    assert_eq!(training.batch_size, 64);

    let lr = training.lr_schedule.as_ref().unwrap();
    assert_eq!(lr.kind, "cosine");
    match lr.extra.get("warmup_steps").unwrap() {
        ConfigValue::Int(500) => {}
        other => panic!("expected Int(500), got {other:?}"),
    }

    let clip = training.grad_clip.as_ref().unwrap();
    assert_eq!(clip.kind, "norm");
}

// Inference config lowering

#[test]
fn test_lower_inference() {
    let src = r#"
        @inference {
            model: Forward;
            quantization: {
                mode: "int8";
            }
            generation: {
                strategy: "top_p";
                temperature: 0.9;
            }
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();

    let inf = ir.inference.as_ref().unwrap();
    assert_eq!(inf.model_graph, "Forward");

    let quant = inf.quantization.as_ref().unwrap();
    match quant.get("mode").unwrap() {
        ConfigValue::Str(s) => assert_eq!(s, "int8"),
        other => panic!("expected Str, got {other:?}"),
    }

    let gen = inf.generation.as_ref().unwrap();
    match gen.get("strategy").unwrap() {
        ConfigValue::Str(s) => assert_eq!(s, "top_p"),
        other => panic!("expected Str, got {other:?}"),
    }
    match gen.get("temperature").unwrap() {
        ConfigValue::Float(t) => assert!((t - 0.9).abs() < 1e-10),
        other => panic!("expected Float, got {other:?}"),
    }
}

// Full program lowering (TinyGPT example)

#[test]
fn test_lower_tiny_gpt() {
    let src = include_str!("../../../examples/tiny_gpt.sw");
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();

    // Metadata
    assert_eq!(ir.metadata.get("name").unwrap(), "TinyGPT");
    assert_eq!(ir.metadata.get("version").unwrap(), "0.1.0");

    // Config with constant folding
    match ir.config.get("d_ff").unwrap() {
        ConfigValue::Int(1024) => {} // 256 * 4
        other => panic!("expected d_ff=1024, got {other:?}"),
    }
    match ir.config.get("vocab_size").unwrap() {
        ConfigValue::Int(50257) => {}
        other => panic!("expected vocab_size=50257, got {other:?}"),
    }

    // Types
    assert!(ir.type_aliases.contains_key("TokenIds"));
    assert!(ir.type_aliases.contains_key("Embeddings"));
    assert!(ir.type_aliases.contains_key("Hidden"));
    assert!(ir.type_aliases.contains_key("Logits"));

    // Graph
    assert_eq!(ir.graphs.len(), 1);
    let graph = &ir.graphs[0];
    assert_eq!(graph.name, "Forward");
    assert!(graph.inputs.len() >= 1);
    assert!(graph.outputs.len() >= 1);
    assert!(graph.params.len() >= 4); // wte, wpe, ln_f_weight, ln_f_bias

    // Training
    let training = ir.training.as_ref().unwrap();
    assert_eq!(training.model_graph, "Forward");
    assert_eq!(training.loss, "cross_entropy");
    assert_eq!(training.optimizer.kind, "AdamW");
    assert_eq!(training.epochs, 20);
    assert_eq!(training.batch_size, 64);
    assert_eq!(training.accumulation_steps, 4);

    // Inference
    let inf = ir.inference.as_ref().unwrap();
    assert_eq!(inf.model_graph, "Forward");
    assert!(inf.quantization.is_some());
    assert!(inf.generation.is_some());

    // Graph dump for visual inspection
    let dump = graph.dump();
    assert!(dump.contains("Forward"));
    assert!(dump.contains("nodes"));
}

// Graph dump formatting

#[test]
fn test_graph_dump() {
    let src = r#"
        @graph Simple {
            input x: Tensor<[Batch, 256], f32>;
            node y {
                op: relu(x);
            };
            output y;
        }
    "#;
    let ast = parse(src).unwrap();
    let ir = lower(&ast).unwrap();
    let dump = ir.graphs[0].dump();

    assert!(dump.contains("Simple"));
    assert!(dump.contains("relu"));
    assert!(dump.contains("identity"));
    assert!(dump.contains("inputs:"));
    assert!(dump.contains("outputs:"));
}
