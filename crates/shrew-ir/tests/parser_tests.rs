// Integration tests for the .sw parser

use shrew_ir::ast::*;
use shrew_ir::parse;

// @model

#[test]
fn test_parse_model_block() {
    let src = r#"
        @model {
            name: "GPT-2";
            version: "1.0";
            author: "OpenAI";
        }
    "#;
    let program = parse(src).unwrap();
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        TopLevel::Metadata(m) => {
            assert_eq!(m.fields.len(), 3);
            assert_eq!(m.fields[0].key, "name");
            match &m.fields[0].value {
                Literal::Str(s, _) => assert_eq!(s, "GPT-2"),
                _ => panic!("expected string literal"),
            }
            assert_eq!(m.fields[1].key, "version");
            assert_eq!(m.fields[2].key, "author");
        }
        _ => panic!("expected Metadata"),
    }
}

// @config

#[test]
fn test_parse_config_block() {
    let src = r#"
        @config {
            d_model: 768;
            n_heads: 12;
            n_layers: 6;
            dropout: 0.1;
            vocab_size: 50257;
        }
    "#;
    let program = parse(src).unwrap();
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        TopLevel::Config(c) => {
            assert_eq!(c.fields.len(), 5);
            assert_eq!(c.fields[0].key, "d_model");
            match &c.fields[0].value {
                Expr::Int(768, _) => {}
                _ => panic!("expected int 768"),
            }
            assert_eq!(c.fields[3].key, "dropout");
            match &c.fields[3].value {
                Expr::Float(f, _) => assert!((f - 0.1).abs() < 1e-10),
                _ => panic!("expected float"),
            }
        }
        _ => panic!("expected Config"),
    }
}

#[test]
fn test_parse_config_with_arithmetic() {
    let src = r#"
        @config {
            d_ff: 768 * 4;
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Config(c) => {
            assert_eq!(c.fields[0].key, "d_ff");
            match &c.fields[0].value {
                Expr::Binary { op: BinOp::Mul, .. } => {}
                _ => panic!("expected binary mul expression"),
            }
        }
        _ => panic!("expected Config"),
    }
}

// @import

#[test]
fn test_parse_import() {
    let src = r#"@import "layers/attention.sw" as attn;"#;
    let program = parse(src).unwrap();
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        TopLevel::Import(i) => {
            assert_eq!(i.path, "layers/attention.sw");
            assert_eq!(i.alias.as_deref(), Some("attn"));
        }
        _ => panic!("expected Import"),
    }
}

#[test]
fn test_parse_import_no_alias() {
    let src = r#"@import "base.sw";"#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Import(i) => {
            assert_eq!(i.path, "base.sw");
            assert!(i.alias.is_none());
        }
        _ => panic!("expected Import"),
    }
}

// @types

#[test]
fn test_parse_types_block() {
    let src = r#"
        @types {
            type Hidden = Tensor<[Batch, 768], f32>;
            type Logits = Tensor<[Batch, 50257], f32>;
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Types(t) => {
            assert_eq!(t.defs.len(), 2);
            assert_eq!(t.defs[0].name, "Hidden");
            match &t.defs[0].ty {
                TypeExpr::Tensor { dims, dtype, .. } => {
                    assert_eq!(dims.len(), 2);
                    match &dims[0] {
                        Dimension::Named(n, _) => assert_eq!(n, "Batch"),
                        _ => panic!("expected named dim"),
                    }
                    match &dims[1] {
                        Dimension::Concrete(768, _) => {}
                        _ => panic!("expected concrete dim 768"),
                    }
                    assert_eq!(*dtype, DTypeKind::F32);
                }
                _ => panic!("expected Tensor type"),
            }
        }
        _ => panic!("expected Types"),
    }
}

#[test]
fn test_parse_type_with_dynamic_dims() {
    let src = r#"
        @types {
            type Input = Tensor<[?, ?, 3], u8>;
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Types(t) => match &t.defs[0].ty {
            TypeExpr::Tensor { dims, dtype, .. } => {
                assert_eq!(dims.len(), 3);
                assert!(matches!(dims[0], Dimension::Dynamic(_)));
                assert!(matches!(dims[1], Dimension::Dynamic(_)));
                assert!(matches!(dims[2], Dimension::Concrete(3, _)));
                assert_eq!(*dtype, DTypeKind::U8);
            }
            _ => panic!("expected Tensor type"),
        },
        _ => panic!("expected Types"),
    }
}

// @graph

#[test]
fn test_parse_graph_basic() {
    let src = r#"
        @graph Forward(x: Tensor<[Batch, SeqLen, 768], f32>) -> Tensor<[Batch, SeqLen, 768], f32> {
            input x: Tensor<[Batch, SeqLen, 768], f32>;

            param W: Tensor<[768, 768], f32> {
                init: "normal(0, 0.02)";
                frozen: false;
            };

            node h {
                op: matmul(x, W);
            };

            output h;
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Graph(g) => {
            assert_eq!(g.name, "Forward");
            assert_eq!(g.params.len(), 1);
            assert_eq!(g.params[0].name, "x");
            assert!(g.return_type.is_some());

            // Check body statements
            assert_eq!(g.body.len(), 4);
            assert!(matches!(g.body[0], GraphStmt::Input(_)));
            assert!(matches!(g.body[1], GraphStmt::Param(_)));
            assert!(matches!(g.body[2], GraphStmt::Node(_)));
            assert!(matches!(g.body[3], GraphStmt::Output(_)));

            // Check param attrs
            match &g.body[1] {
                GraphStmt::Param(p) => {
                    assert_eq!(p.name, "W");
                    assert_eq!(p.attrs.len(), 2);
                    assert_eq!(p.attrs[0].key, "init");
                    assert_eq!(p.attrs[1].key, "frozen");
                }
                _ => panic!(),
            }

            // Check node
            match &g.body[2] {
                GraphStmt::Node(n) => {
                    assert_eq!(n.name, "h");
                    assert_eq!(n.stmts.len(), 1);
                    match &n.stmts[0] {
                        NodeStmt::Op(Expr::Call { func, args, .. }, _) => {
                            assert_eq!(func, "matmul");
                            assert_eq!(args.len(), 2);
                        }
                        _ => panic!("expected op with call"),
                    }
                }
                _ => panic!(),
            }
        }
        _ => panic!("expected Graph"),
    }
}

#[test]
fn test_parse_graph_no_params_no_return() {
    let src = r#"
        @graph Simple {
            input x: f32;
            node y {
                op: x + 1;
            };
            output y;
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Graph(g) => {
            assert_eq!(g.name, "Simple");
            assert!(g.params.is_empty());
            assert!(g.return_type.is_none());
            assert_eq!(g.body.len(), 3);
        }
        _ => panic!("expected Graph"),
    }
}

#[test]
fn test_parse_graph_with_assert() {
    let src = r#"
        @graph Checked {
            input x: Tensor<[Batch, 768], f32>;
            @assert Batch > 0, "batch must be positive";
            output x;
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Graph(g) => {
            assert_eq!(g.body.len(), 3);
            match &g.body[1] {
                GraphStmt::Assert(a) => {
                    assert_eq!(a.message.as_deref(), Some("batch must be positive"));
                }
                _ => panic!("expected Assert"),
            }
        }
        _ => panic!("expected Graph"),
    }
}

#[test]
fn test_parse_node_with_hint() {
    let src = r#"
        @graph WithHints {
            node h {
                op: expensive_op(x);
                @hint recompute_in_backward;
            };
            output h;
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Graph(g) => match &g.body[0] {
            GraphStmt::Node(n) => {
                assert_eq!(n.stmts.len(), 2);
                assert!(matches!(
                    n.stmts[1],
                    NodeStmt::Hint(HintKind::RecomputeInBackward, _)
                ));
            }
            _ => panic!(),
        },
        _ => panic!(),
    }
}

// @training

#[test]
fn test_parse_training_block() {
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
                warmup_steps: 1000;
            }
            precision: "fp16";
            epochs: 10;
            batch_size: 64;
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Training(t) => {
            // model, loss, optimizer, lr_schedule, precision, epochs, batch_size
            assert_eq!(t.fields.len(), 7);
            assert!(matches!(t.fields[0], TrainingField::Model(ref s, _) if s == "Forward"));
            assert!(matches!(t.fields[1], TrainingField::Loss(ref s, _) if s == "cross_entropy"));
            match &t.fields[2] {
                TrainingField::Optimizer(fields, _) => {
                    assert_eq!(fields.len(), 3);
                    assert_eq!(fields[0].key, "type");
                    assert_eq!(fields[1].key, "lr");
                }
                _ => panic!("expected Optimizer"),
            }
            match &t.fields[3] {
                TrainingField::LrSchedule(fields, _) => {
                    assert_eq!(fields.len(), 2);
                }
                _ => panic!("expected LrSchedule"),
            }
        }
        _ => panic!("expected Training"),
    }
}

// @inference

#[test]
fn test_parse_inference_block() {
    let src = r#"
        @inference {
            model: Forward;
            quantization: {
                mode: "int8";
            }
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Inference(i) => {
            assert_eq!(i.fields.len(), 2);
            assert!(matches!(i.fields[0], InferenceField::Model(ref s, _) if s == "Forward"));
            match &i.fields[1] {
                InferenceField::Quantization(fields, _) => {
                    assert_eq!(fields.len(), 1);
                    assert_eq!(fields[0].key, "mode");
                }
                _ => panic!("expected Quantization"),
            }
        }
        _ => panic!("expected Inference"),
    }
}

// @metrics

#[test]
fn test_parse_metrics_block() {
    let src = r#"
        @metrics TrainingMetrics {
            track train_loss {
                source: loss;
                aggregate: "mean";
                log_every: 10;
            }
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Metrics(m) => {
            assert_eq!(m.name, "TrainingMetrics");
            assert_eq!(m.defs.len(), 1);
            assert_eq!(m.defs[0].name, "train_loss");
            assert_eq!(m.defs[0].attrs.len(), 3);
        }
        _ => panic!("expected Metrics"),
    }
}

// @logging

#[test]
fn test_parse_logging_block() {
    let src = r#"
        @logging {
            backend: "tensorboard";
            log_dir: "./logs";
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Logging(l) => {
            assert_eq!(l.fields.len(), 2);
            assert_eq!(l.fields[0].key, "backend");
            assert_eq!(l.fields[1].key, "log_dir");
        }
        _ => panic!("expected Logging"),
    }
}

// @visualizations

#[test]
fn test_parse_visualization_block() {
    let src = r#"
        @visualizations {
            plot loss_curve {
                x: "step";
                y: "loss";
            }
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Visualization(v) => {
            assert_eq!(v.plots.len(), 1);
            assert_eq!(v.plots[0].name, "loss_curve");
            assert_eq!(v.plots[0].attrs.len(), 2);
        }
        _ => panic!("expected Visualization"),
    }
}

// Expression parsing

#[test]
fn test_parse_operator_precedence() {
    // 1 + 2 * 3 should parse as 1 + (2 * 3)
    let src = r#"
        @config {
            result: 1 + 2 * 3;
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Config(c) => match &c.fields[0].value {
            Expr::Binary {
                op: BinOp::Add,
                right,
                ..
            } => {
                assert!(matches!(**right, Expr::Binary { op: BinOp::Mul, .. }));
            }
            _ => panic!("expected Add at top"),
        },
        _ => panic!(),
    }
}

#[test]
fn test_parse_unary_neg() {
    let src = r#"
        @config {
            neg: -42;
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Config(c) => match &c.fields[0].value {
            Expr::Unary {
                op: UnaryOp::Neg,
                operand,
                ..
            } => {
                assert!(matches!(**operand, Expr::Int(42, _)));
            }
            _ => panic!("expected unary neg"),
        },
        _ => panic!(),
    }
}

#[test]
fn test_parse_function_call() {
    let src = r#"
        @config {
            result: softmax(logits, dim: 1);
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Config(c) => match &c.fields[0].value {
            Expr::Call { func, args, .. } => {
                assert_eq!(func, "softmax");
                assert_eq!(args.len(), 2);
                assert!(args[0].name.is_none());
                assert_eq!(args[1].name.as_deref(), Some("dim"));
            }
            _ => panic!("expected function call"),
        },
        _ => panic!(),
    }
}

#[test]
fn test_parse_list_expression() {
    let src = r#"
        @config {
            sizes: [128, 256, 512];
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Config(c) => match &c.fields[0].value {
            Expr::List(items, _) => {
                assert_eq!(items.len(), 3);
            }
            _ => panic!("expected list"),
        },
        _ => panic!(),
    }
}

#[test]
fn test_parse_boolean_logic() {
    let src = r#"
        @config {
            cond: true && false || true;
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::Config(c) => {
            // || has lower precedence than &&, so: (true && false) || true
            match &c.fields[0].value {
                Expr::Binary {
                    op: BinOp::Or,
                    left,
                    ..
                } => {
                    assert!(matches!(**left, Expr::Binary { op: BinOp::And, .. }));
                }
                _ => panic!("expected Or at top"),
            }
        }
        _ => panic!(),
    }
}

// @custom_op

#[test]
fn test_parse_custom_op() {
    let src = r#"
        @custom_op RotaryEmbedding {
            signature: (x: Tensor<[B, S, D], f32>, freqs: Tensor<[S, D], f32>) -> Tensor<[B, S, D], f32>;
            impl cpu {
                kernel: "rotary_cpu";
            }
            gradient backward {
                impl cpu {
                    kernel: "rotary_grad_cpu";
                }
            }
        }
    "#;
    let program = parse(src).unwrap();
    match &program.items[0] {
        TopLevel::CustomOp(co) => {
            assert_eq!(co.name, "RotaryEmbedding");
            assert_eq!(co.stmts.len(), 3); // signature, impl, gradient
            match &co.stmts[0] {
                CustomOpStmt::Signature { params, .. } => {
                    assert_eq!(params.len(), 2);
                    assert_eq!(params[0].name, "x");
                    assert_eq!(params[1].name, "freqs");
                }
                _ => panic!("expected signature"),
            }
        }
        _ => panic!("expected CustomOp"),
    }
}

// Full .sw program (multiple blocks)

#[test]
fn test_parse_full_program() {
    let src = r#"
        // A small GPT-style model
        @model {
            name: "TinyGPT";
            version: "0.1";
        }

        @config {
            d_model: 256;
            n_heads: 4;
            n_layers: 2;
            vocab_size: 1000;
        }

        @types {
            type Hidden = Tensor<[Batch, SeqLen, 256], f32>;
        }

        @graph Forward {
            input tokens: Tensor<[Batch, SeqLen], i64>;

            node embeddings {
                op: Embedding(tokens, vocab_size: 1000, d_model: 256);
            };

            output embeddings;
        }

        @training {
            model: Forward;
            loss: cross_entropy;
            optimizer: {
                type: "Adam";
                lr: 1e-3;
            }
            epochs: 5;
        }

        @logging {
            backend: "stdout";
        }
    "#;
    let program = parse(src).unwrap();
    assert_eq!(program.items.len(), 6);
    assert!(matches!(program.items[0], TopLevel::Metadata(_)));
    assert!(matches!(program.items[1], TopLevel::Config(_)));
    assert!(matches!(program.items[2], TopLevel::Types(_)));
    assert!(matches!(program.items[3], TopLevel::Graph(_)));
    assert!(matches!(program.items[4], TopLevel::Training(_)));
    assert!(matches!(program.items[5], TopLevel::Logging(_)));
}

// Error cases

#[test]
fn test_parse_error_unknown_directive() {
    let result = parse("@foobar {}");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(format!("{err}").contains("unknown directive"));
}

#[test]
fn test_parse_error_unterminated_string() {
    let result = parse(r#"@model { name: "oops; }"#);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(format!("{err}").contains("unterminated string"));
}

#[test]
fn test_parse_error_unexpected_token() {
    let result = parse("@model { 42: value; }");
    assert!(result.is_err());
}

// Comments

#[test]
fn test_comments_preserved_semantics() {
    let src = r#"
        // This is a line comment
        @model {
            /* block comment in the middle */
            name: "test";
        }
    "#;
    let program = parse(src).unwrap();
    assert_eq!(program.items.len(), 1);
    match &program.items[0] {
        TopLevel::Metadata(m) => {
            assert_eq!(m.fields.len(), 1);
        }
        _ => panic!(),
    }
}
#[test]
fn test_tiny_gpt_example() {
    let src = include_str!("../../../examples/tiny_gpt.sw");
    let program = shrew_ir::parse(src).unwrap();
    // @import, @model, @config, @types, @graph, @custom_op, @training, @inference, @metrics, @logging, @visualizations
    assert_eq!(program.items.len(), 11);
}
