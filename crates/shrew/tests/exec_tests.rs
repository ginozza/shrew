// Executor tests — Verifies .sw files can be parsed and executed end-to-end

use std::collections::HashMap;

use shrew::exec::{load_program, load_trainer, Executor, RuntimeConfig};
use shrew::prelude::*;

// Helper: build executor from .sw source

fn make_executor(src: &str, config: RuntimeConfig) -> Executor<CpuBackend> {
    load_program::<CpuBackend>(src, CpuDevice, config).expect("failed to load program")
}

// Basic graph execution

#[test]
fn test_exec_identity_graph() {
    let src = r#"
@model { name: "Identity"; }
@graph Forward {
    input x: Tensor<[2, 3], f64>;
    node out { op: x; };
    output out;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x.clone());

    let result = exec.run("Forward", &inputs).unwrap();
    let out = result.get("out").expect("no output 'out'");
    assert_eq!(out.dims(), &[2, 3]);

    let out_data = out.to_f64_vec().unwrap();
    assert_eq!(out_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_exec_add_graph() {
    let src = r#"
@model { name: "AddNet"; }
@graph Forward {
    input a: Tensor<[3], f64>;
    input b: Tensor<[3], f64>;
    node c { op: a + b; };
    output c;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &CpuDevice).unwrap();
    let b = CpuTensor::from_f64_slice(&[10.0, 20.0, 30.0], 3, DType::F64, &CpuDevice).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), a);
    inputs.insert("b".to_string(), b);

    let result = exec.run("Forward", &inputs).unwrap();
    let c = result.get("c").expect("no output 'c'");
    let data = c.to_f64_vec().unwrap();
    assert_eq!(data, vec![11.0, 22.0, 33.0]);
}

#[test]
fn test_exec_binary_ops() {
    let src = r#"
@model { name: "BinaryOps"; }
@graph Forward {
    input x: Tensor<[4], f64>;
    input y: Tensor<[4], f64>;
    node sum { op: x + y; };
    node diff { op: x - y; };
    node prod { op: x * y; };
    node quot { op: x / y; };
    output quot;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x =
        CpuTensor::from_f64_slice(&[10.0, 20.0, 30.0, 40.0], 4, DType::F64, &CpuDevice).unwrap();
    let y = CpuTensor::from_f64_slice(&[2.0, 4.0, 5.0, 8.0], 4, DType::F64, &CpuDevice).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);
    inputs.insert("y".to_string(), y);

    let result = exec.run("Forward", &inputs).unwrap();
    let q = result.get("quot").unwrap();
    let data = q.to_f64_vec().unwrap();
    assert_eq!(data, vec![5.0, 5.0, 6.0, 5.0]);
}

#[test]
fn test_exec_unary_relu() {
    let src = r#"
@model { name: "Relu"; }
@graph Forward {
    input x: Tensor<[5], f64>;
    node y { op: relu(x); };
    output y;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x =
        CpuTensor::from_f64_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0], 5, DType::F64, &CpuDevice).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let y = result.get("y").unwrap();
    let data = y.to_f64_vec().unwrap();
    assert_eq!(data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_exec_matmul() {
    let src = r#"
@model { name: "MatMul"; }
@graph Forward {
    input a: Tensor<[2, 3], f64>;
    input b: Tensor<[3, 2], f64>;
    node c { op: matmul(a, b); };
    output c;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    // a = [[1,0,0],[0,1,0]] (2x3 identity-ish)
    let a = CpuTensor::from_f64_slice(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        (2, 3),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();
    // b = [[1,2],[3,4],[5,6]] (3x2)
    let b = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        (3, 2),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), a);
    inputs.insert("b".to_string(), b);

    let result = exec.run("Forward", &inputs).unwrap();
    let c = result.get("c").unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    let data = c.to_f64_vec().unwrap();
    // [1,0,0]·[1,3,5] = 1, [1,0,0]·[2,4,6] = 2, [0,1,0]·[1,3,5] = 3, [0,1,0]·[2,4,6] = 4
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_exec_softmax() {
    let src = r#"
@model { name: "Softmax"; }
@graph Forward {
    input x: Tensor<[3], f64>;
    node y { op: softmax(x, dim: 0); };
    output y;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let y = result.get("y").unwrap();
    let data = y.to_f64_vec().unwrap();

    // Softmax should sum to 1
    let sum: f64 = data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "softmax sum = {}", sum);

    // Values should be in ascending order
    assert!(data[0] < data[1]);
    assert!(data[1] < data[2]);
}

#[test]
fn test_exec_transpose() {
    let src = r#"
@model { name: "Transpose"; }
@graph Forward {
    input x: Tensor<[2, 3], f64>;
    node y { op: transpose(x); };
    output y;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let y = result.get("y").unwrap();
    assert_eq!(y.dims(), &[3, 2]);
}

// Parameter initialization

#[test]
fn test_exec_param_init_zeros() {
    let src = r#"
@model { name: "ParamTest"; }
@graph Forward {
    input x: Tensor<[3], f64>;
    param w: Tensor<[3], f64> { init: "zeros"; };
    node y { op: x + w; };
    output y;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    // With zero params, output should equal input
    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let y = result.get("y").unwrap();
    let data = y.to_f64_vec().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_exec_param_init_ones() {
    let src = r#"
@model { name: "ParamOnes"; }
@graph Forward {
    input x: Tensor<[3], f64>;
    param w: Tensor<[3], f64> { init: "ones"; };
    node y { op: x * w; };
    output y;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(&[2.0, 3.0, 4.0], 3, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let y = result.get("y").unwrap();
    let data = y.to_f64_vec().unwrap();
    assert_eq!(data, vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_exec_param_normal_init() {
    let src = r#"
@model { name: "ParamNormal"; }
@graph Forward {
    input x: Tensor<[4, 8], f64>;
    param w: Tensor<[8, 4], f64> { init: "normal(0, 0.02)"; };
    node y { op: matmul(x, w); };
    output y;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x = CpuTensor::rand((4, 8), DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let y = result.get("y").unwrap();
    assert_eq!(y.dims(), &[4, 4]);
}

// Embedding execution

#[test]
fn test_exec_embedding() {
    let src = r#"
@model { name: "EmbTest"; }
@graph Forward {
    input tokens: Tensor<[3], i64>;
    param wte: Tensor<[10, 4], f64> { init: "ones"; };
    node emb { op: embedding(tokens, wte); };
    output emb;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let tokens = CpuTensor::from_f64_slice(&[0.0, 1.0, 2.0], 3, DType::I64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("tokens".to_string(), tokens);

    let result = exec.run("Forward", &inputs).unwrap();
    let emb = result.get("emb").unwrap();
    assert_eq!(emb.dims(), &[3, 4]);

    // All ones init → every embedding vector is [1, 1, 1, 1]
    let data = emb.to_f64_vec().unwrap();
    assert!(data.iter().all(|&v| (v - 1.0).abs() < 1e-10));
}

// LayerNorm execution

#[test]
fn test_exec_layernorm() {
    let src = r#"
@model { name: "LNTest"; }
@graph Forward {
    input x: Tensor<[2, 4], f64>;
    param ln_w: Tensor<[4], f64> { init: "ones"; };
    param ln_b: Tensor<[4], f64> { init: "zeros"; };
    node y { op: layer_norm(x, ln_w, ln_b, eps: 1e-5); };
    output y;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        (2, 4),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let y = result.get("y").unwrap();
    assert_eq!(y.dims(), &[2, 4]);

    // LayerNorm with weight=1, bias=0 → zero mean, unit variance per row
    let data = y.to_f64_vec().unwrap();
    // First row mean should be ~0
    let row1_mean: f64 = data[0..4].iter().sum::<f64>() / 4.0;
    assert!(row1_mean.abs() < 1e-6, "row1 mean = {}", row1_mean);
}

// Chain of operations

#[test]
fn test_exec_multi_node_chain() {
    let src = r#"
@model { name: "Chain"; }
@graph Forward {
    input x: Tensor<[4], f64>;
    node a { op: relu(x); };
    node b { op: a + a; };
    node c { op: sigmoid(b); };
    output c;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(&[-1.0, 0.0, 1.0, 2.0], 4, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let c = result.get("c").unwrap();
    let data = c.to_f64_vec().unwrap();

    // relu([-1,0,1,2]) = [0,0,1,2]
    // + itself: [0,0,2,4]
    // sigmoid: [0.5, 0.5, sigmoid(2), sigmoid(4)]
    assert!((data[0] - 0.5).abs() < 1e-6);
    assert!((data[1] - 0.5).abs() < 1e-6);
    assert!(data[2] > 0.8); // sigmoid(2) ≈ 0.88
    assert!(data[3] > 0.98); // sigmoid(4) ≈ 0.98
}

// Matmul + transpose (like TinyGPT's logits computation)

#[test]
fn test_exec_matmul_transpose() {
    let src = r#"
@model { name: "MatMulT"; }
@graph Forward {
    input x: Tensor<[2, 4], f64>;
    param w: Tensor<[3, 4], f64> { init: "ones"; };
    node logits { op: matmul(x, transpose(w)); };
    output logits;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(
        &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        (2, 4),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let logits = result.get("logits").unwrap();
    assert_eq!(logits.dims(), &[2, 3]);

    // w is all-ones [3,4], w^T is [4,3]
    // x[0]·w^T = [4.0, 4.0, 4.0], x[1]·w^T = [8.0, 8.0, 8.0]
    let data = logits.to_f64_vec().unwrap();
    assert_eq!(data, vec![4.0, 4.0, 4.0, 8.0, 8.0, 8.0]);
}

// Symbolic dimension resolution

#[test]
fn test_exec_symbolic_dims() {
    let src = r#"
@model { name: "SymDim"; }
@config { d_model: 8; }
@graph Forward {
    input x: Tensor<[Batch, d_model], f64>;
    param w: Tensor<[d_model, d_model], f64> { init: "zeros"; };
    node y { op: matmul(x, w); };
    output y;
}
"#;
    let config = RuntimeConfig::default()
        .with_dtype(DType::F64)
        .set_dim("Batch", 3);
    let exec = make_executor(src, config);

    let x = CpuTensor::rand((3, 8), DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let y = result.get("y").unwrap();
    assert_eq!(y.dims(), &[3, 8]);
}

// Sum reduction

#[test]
fn test_exec_sum_reduction() {
    let src = r#"
@model { name: "Sum"; }
@graph Forward {
    input x: Tensor<[2, 3], f64>;
    node y { op: sum(x, dim: 1); };
    output y;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let y = result.get("y").unwrap();
    let data = y.to_f64_vec().unwrap();
    // sum([1,2,3])=6, sum([4,5,6])=15
    assert_eq!(data, vec![6.0, 15.0]);
}

// Mean reduction

#[test]
fn test_exec_mean_reduction() {
    let src = r#"
@model { name: "Mean"; }
@graph Forward {
    input x: Tensor<[2, 4], f64>;
    node y { op: mean(x, dim: 1); };
    output y;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        (2, 4),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let y = result.get("y").unwrap();
    let data = y.to_f64_vec().unwrap();
    assert_eq!(data, vec![2.5, 25.0]);
}

// Constant nodes

#[test]
fn test_exec_constant() {
    let src = r#"
@model { name: "Const"; }
@config { scale: 2; }
@graph Forward {
    input x: Tensor<[3], f64>;
    node doubled { op: x + x; };
    output doubled;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(&[5.0, 10.0, 15.0], 3, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let out = result.get("doubled").unwrap();
    let data = out.to_f64_vec().unwrap();
    assert_eq!(data, vec![10.0, 20.0, 30.0]);
}

// Graph not found

#[test]
fn test_exec_graph_not_found() {
    let src = r#"
@model { name: "Test"; }
@graph Forward {
    input x: Tensor<[3], f64>;
    output x;
}
"#;
    let config = RuntimeConfig::default();
    let exec = make_executor(src, config);

    let inputs = HashMap::new();
    let result = exec.run("NonExistent", &inputs);
    assert!(result.is_err());
}

// Dropout (eval mode = pass-through)

#[test]
fn test_exec_dropout_eval() {
    let src = r#"
@model { name: "DropoutTest"; }
@graph Forward {
    input x: Tensor<[2, 3], f64>;
    node y { op: dropout(x, p: 0.5); };
    output y;
}
"#;
    // Eval mode (training=false) → dropout is a no-op
    let config = RuntimeConfig::default()
        .with_dtype(DType::F64)
        .with_training(false);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let y = result.get("y").unwrap();
    let data = y.to_f64_vec().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

// Linear layer execution

#[test]
fn test_exec_linear() {
    let src = r#"
@model { name: "LinearTest"; }
@graph Forward {
    input x: Tensor<[2, 4], f64>;
    param w: Tensor<[3, 4], f64> { init: "ones"; };
    node y { op: linear(x, w, bias: false); };
    output y;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(
        &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        (2, 4),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let y = result.get("y").unwrap();
    assert_eq!(y.dims(), &[2, 3]);
    // x @ w^T: [1,1,1,1] @ ones^T_3x4 = [4,4,4], [2,2,2,2] @ ... = [8,8,8]
    let data = y.to_f64_vec().unwrap();
    assert_eq!(data, vec![4.0, 4.0, 4.0, 8.0, 8.0, 8.0]);
}

// Gelu activation

#[test]
fn test_exec_gelu() {
    let src = r#"
@model { name: "GeluTest"; }
@graph Forward {
    input x: Tensor<[3], f64>;
    node y { op: gelu(x); };
    output y;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(&[0.0, 1.0, -1.0], 3, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let y = result.get("y").unwrap();
    let data = y.to_f64_vec().unwrap();
    // GELU(0) = 0
    assert!((data[0]).abs() < 1e-6);
    // GELU(1) > 0
    assert!(data[1] > 0.8);
    // GELU(-1) < 0
    assert!(data[2] < 0.0);
}

// Multiple unary activations

#[test]
fn test_exec_activation_chain() {
    let src = r#"
@model { name: "ActChain"; }
@graph Forward {
    input x: Tensor<[4], f64>;
    node a { op: exp(x); };
    node b { op: log(a); };
    output b;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], 4, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let b = result.get("b").unwrap();
    let data = b.to_f64_vec().unwrap();
    // log(exp(x)) = x
    for (i, &v) in data.iter().enumerate() {
        assert!(
            (v - (i as f64 + 1.0)).abs() < 1e-10,
            "log(exp({})) = {}",
            i + 1,
            v
        );
    }
}

// Full pipeline from .sw source text

#[test]
fn test_exec_full_pipeline_simple_net() {
    // A simple 2-layer network: x → linear → relu → linear → output
    let src = r#"
@model { name: "SimpleNet"; version: "1.0"; }
@config { d_in: 4; d_hidden: 8; d_out: 2; }

@graph Forward {
    input x: Tensor<[Batch, 4], f64>;
    param w1: Tensor<[8, 4], f64>  { init: "normal(0, 0.1)"; };
    param w2: Tensor<[2, 8], f64>  { init: "normal(0, 0.1)"; };

    node h  { op: linear(x, w1, bias: false); };
    node ha { op: relu(h); };
    node y  { op: linear(ha, w2, bias: false); };
    output y;
}
"#;
    let config = RuntimeConfig::default()
        .with_dtype(DType::F64)
        .set_dim("Batch", 3);
    let exec = make_executor(src, config);

    let x = CpuTensor::rand((3, 4), DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let y = result.get("y").unwrap();
    assert_eq!(y.dims(), &[3, 2]);
}

// Autograd through executed graph

#[test]
fn test_exec_backward_through_graph() {
    let src = r#"
@model { name: "GradTest"; }
@graph Forward {
    input x: Tensor<[2, 3], f64>;
    param w: Tensor<[3, 1], f64> { init: "ones"; };
    node y { op: matmul(x, w); };
    node loss { op: mean(y, dim: 0); };
    output loss;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        DType::F64,
        &CpuDevice,
    )
    .unwrap()
    .set_variable();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let loss = result.get("loss").unwrap();

    // The loss should be a scalar (or near-scalar)
    // mean of [1+2+3, 4+5+6] = mean of [6, 15] = [10.5]
    let loss_val = loss.to_f64_vec().unwrap();
    assert!((loss_val[0] - 10.5).abs() < 1e-6);

    // Should be able to call backward
    let sum_loss = loss.sum_all().unwrap();
    let _grads = sum_loss.backward().unwrap();
}

// Trainer creation

#[test]
fn test_trainer_from_program() {
    let src = r#"
@model { name: "TrainTest"; }
@graph Forward {
    input x: Tensor<[4, 3], f64>;
    param w: Tensor<[3, 2], f64> { init: "normal(0, 0.1)"; };
    node y { op: matmul(x, w); };
    output y;
}

@training {
    model: Forward;
    loss: mse;
    optimizer: { type: "SGD"; lr: 0.01; }
    epochs: 5;
    batch_size: 4;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let trainer =
        load_trainer::<CpuBackend>(src, CpuDevice, config).expect("failed to create trainer");

    assert_eq!(trainer.model_graph_name(), "Forward");
    assert_eq!(trainer.loss_fn_name(), "mse");
    assert_eq!(trainer.epochs(), 5);
    assert_eq!(trainer.batch_size, 4);
}

// Training loop (simple regression)

#[test]
fn test_trainer_simple_regression() {
    let src = r#"
@model { name: "Regression"; }
@graph Forward {
    input x: Tensor<[4, 1], f64>;
    param w: Tensor<[1, 1], f64> { init: "ones"; };
    node y { op: matmul(x, w); };
    output y;
}

@training {
    model: Forward;
    loss: mse;
    optimizer: { type: "SGD"; lr: 0.01; }
    epochs: 50;
    batch_size: 4;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let mut trainer =
        load_trainer::<CpuBackend>(src, CpuDevice, config).expect("failed to create trainer");

    // Target: y = 2*x
    let x =
        CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (4, 1), DType::F64, &CpuDevice).unwrap();
    let targets =
        CpuTensor::from_f64_slice(&[2.0, 4.0, 6.0, 8.0], (4, 1), DType::F64, &CpuDevice).unwrap();

    let mut batch = HashMap::new();
    batch.insert("x".to_string(), x);
    batch.insert("targets".to_string(), targets);

    let result = trainer.train(&[batch], "targets").unwrap();
    // Loss should decrease over epochs
    assert!(
        result.final_loss < result.epochs[0].loss,
        "Loss didn't decrease: first={}, final={}",
        result.epochs[0].loss,
        result.final_loss
    );
}

// RuntimeConfig builder

#[test]
fn test_runtime_config() {
    let config = RuntimeConfig::default()
        .set_dim("Batch", 32)
        .set_dim("SeqLen", 128)
        .with_training(true)
        .with_dtype(DType::F32);

    assert_eq!(config.dims["Batch"], 32);
    assert_eq!(config.dims["SeqLen"], 128);
    assert!(config.training);
    assert_eq!(config.default_dtype, DType::F32);
}

// End-to-end integration: MLP classifier

/// A small 2-layer MLP that can be parsed, lowered, and executed end-to-end.
#[test]
fn test_exec_e2e_mlp() {
    let src = r#"
@model { name: "MLP"; }
@graph Forward {
    input x: Tensor<[2, 4], f32>;

    param w1: Tensor<[4, 8], f32> { init: "normal(0, 0.1)"; };
    param b1: Tensor<[8], f32>    { init: "zeros"; };
    param w2: Tensor<[8, 3], f32> { init: "normal(0, 0.1)"; };
    param b2: Tensor<[3], f32>    { init: "zeros"; };

    node h1     { op: matmul(x, w1) + b1; };
    node a1     { op: relu(h1); };
    node logits { op: matmul(a1, w2) + b2; };
    node probs  { op: softmax(logits); };

    output probs;
}
"#;

    let config = RuntimeConfig::default().with_dtype(DType::F32);
    let exec = make_executor(src, config);

    let x = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        (2, 4),
        DType::F32,
        &CpuDevice,
    )
    .unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = exec.run("Forward", &inputs).unwrap();
    let probs = result.get("probs").expect("no output 'probs'");

    // Output shape should be [2, 3]
    assert_eq!(probs.dims(), &[2, 3]);

    // Each row should sum to ~1.0 (softmax)
    let data = probs.to_f64_vec().unwrap();
    let row1_sum: f64 = data[0..3].iter().sum();
    let row2_sum: f64 = data[3..6].iter().sum();
    assert!((row1_sum - 1.0).abs() < 1e-5, "row 1 sum = {row1_sum}");
    assert!((row2_sum - 1.0).abs() < 1e-5, "row 2 sum = {row2_sum}");

    // All probabilities non-negative
    for &v in &data {
        assert!(v >= 0.0, "negative probability: {v}");
    }
}

/// Verify the tiny_gpt.sw example can at least be parsed and lowered to IR.
#[test]
fn test_tiny_gpt_parse_and_lower() {
    // Read a subset of tiny_gpt that the parser can handle
    // (skip @import, @custom_op, @metrics, @logging, @visualizations)
    let src = r#"
@model {
    name: "TinyGPT";
    version: "0.1.0";
}

@config {
    vocab_size: 50257;
    d_model: 256;
    n_heads: 4;
    n_layers: 4;
    max_seq_len: 512;
    dropout: 0.1;
    eps: 1e-5;
}

@graph Forward {
    input tokens: Tensor<[Batch, SeqLen], i64>;

    param wte: Tensor<[50257, 256], f32> { init: "normal(0, 0.02)"; };
    param wpe: Tensor<[512, 256], f32>   { init: "normal(0, 0.02)"; };
    param ln_f_weight: Tensor<[256], f32> { init: "ones"; };
    param ln_f_bias: Tensor<[256], f32>   { init: "zeros"; };

    node tok_emb { op: embedding(tokens, wte); };
    node h       { op: tok_emb; };
    node ln_out  { op: layer_norm(h, ln_f_weight, ln_f_bias, eps: 1e-5); };
    node logits  { op: matmul(ln_out, transpose(wte)); };

    output logits;
}

@training {
    model: Forward;
    loss: cross_entropy;
    optimizer: {
        type: "AdamW";
        lr: 3e-4;
    }
    epochs: 20;
    batch_size: 64;
}
"#;

    use shrew_ir::validate::validate;
    use shrew_ir::{lower, parse};

    let ast = parse(src).expect("parse failed");
    let ir = lower(&ast).expect("lowering failed");
    validate(&ir).expect("validation failed");

    // Should have a Forward graph
    assert!(
        ir.graphs.iter().any(|g| g.name == "Forward"),
        "missing Forward graph"
    );

    let fwd = ir.graphs.iter().find(|g| g.name == "Forward").unwrap();

    // Should have params wte, wpe, ln_f_weight, ln_f_bias
    assert!(
        fwd.params.len() >= 4,
        "expected >= 4 params, got {}",
        fwd.params.len()
    );

    // Should have at least 1 output
    assert!(!fwd.outputs.is_empty());
}

// Checkpoint save/load tests

#[test]
fn test_checkpoint_executor_roundtrip() {
    let src = r#"
@model { name: "CheckpointTest"; }
@graph Forward {
    input x: Tensor<[2, 3], f64>;
    param w: Tensor<[3, 4], f64> { init: "normal(0, 0.1)"; };
    param b: Tensor<[1, 4], f64> { init: "zeros"; };
    node h { op: matmul(x, w); };
    node out { op: add(h, b); };
    output out;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config.clone());

    // Save to memory
    let named = exec.named_params();
    assert_eq!(named.len(), 2);
    assert!(named.iter().any(|(k, _)| k == "Forward/w"));
    assert!(named.iter().any(|(k, _)| k == "Forward/b"));

    let bytes = shrew::checkpoint::to_bytes(&named).unwrap();

    // Create a new executor (fresh random params)
    let mut exec2 = make_executor(src, config);

    // Load saved params
    let loaded = shrew::checkpoint::from_bytes::<CpuBackend>(&bytes, &CpuDevice).unwrap();
    for (key, tensor) in &loaded {
        exec2.set_param_by_key(key, tensor.clone());
    }

    // Verify params match original
    let p1_orig: Vec<f64> = named
        .iter()
        .find(|(k, _)| k == "Forward/w")
        .unwrap()
        .1
        .to_f64_vec()
        .unwrap();
    let p1_after: Vec<f64> = exec2
        .named_params()
        .iter()
        .find(|(k, _)| k == "Forward/w")
        .unwrap()
        .1
        .to_f64_vec()
        .unwrap();
    assert_eq!(p1_orig, p1_after);

    // Verify execution produces same outputs
    let x = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let r1 = exec.run("Forward", &inputs).unwrap();
    let r2 = exec2.run("Forward", &inputs).unwrap();

    let out1 = r1.get("out").unwrap().to_f64_vec().unwrap();
    let out2 = r2.get("out").unwrap().to_f64_vec().unwrap();
    assert_eq!(out1, out2);
}

#[test]
fn test_checkpoint_file_save_load() {
    let src = r#"
@model { name: "FileTest"; }
@graph Forward {
    input x: Tensor<[1, 2], f64>;
    param w: Tensor<[2, 3], f64> { init: "normal(0, 0.1)"; };
    node out { op: matmul(x, w); };
    output out;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config.clone());

    let path = std::env::temp_dir().join("shrew_exec_ckpt_test.shrew");
    shrew::checkpoint::save(&path, &exec).unwrap();

    let mut exec2 = make_executor(src, config);
    let loaded = shrew::checkpoint::load(&path, &mut exec2).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(loaded, 1); // 1 param (w)

    // Verify same output
    let x = CpuTensor::from_f64_slice(&[1.0, 2.0], (1, 2), DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let r1 = exec.run("Forward", &inputs).unwrap();
    let r2 = exec2.run("Forward", &inputs).unwrap();
    let o1 = r1.get("out").unwrap().to_f64_vec().unwrap();
    let o2 = r2.get("out").unwrap().to_f64_vec().unwrap();
    assert_eq!(o1, o2);
}

// Mod op test (verifies floor-based modulo)

#[test]
fn test_exec_mod_op() {
    let src = r#"
@model { name: "ModTest"; }
@graph Forward {
    input a: Tensor<[3], f64>;
    input b: Tensor<[3], f64>;
    node out { op: a % b; };
    output out;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    let a = CpuTensor::from_f64_slice(&[7.0, 5.5, -7.0], 3, DType::F64, &CpuDevice).unwrap();
    let b = CpuTensor::from_f64_slice(&[3.0, 2.0, 3.0], 3, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), a);
    inputs.insert("b".to_string(), b);

    let result = exec.run("Forward", &inputs).unwrap();
    let out = result.get("out").unwrap().to_f64_vec().unwrap();
    // 7 % 3 = 1, 5.5 % 2 = 1.5, -7 % 3 = -7 - floor(-7/3)*3 = -7 - (-3)*3 = -7+9 = 2
    assert!(
        (out[0] - 1.0).abs() < 1e-10,
        "7 mod 3 = {}, expected 1",
        out[0]
    );
    assert!(
        (out[1] - 1.5).abs() < 1e-10,
        "5.5 mod 2 = {}, expected 1.5",
        out[1]
    );
    assert!(
        (out[2] - 2.0).abs() < 1e-10,
        "-7 mod 3 = {}, expected 2",
        out[2]
    );
}

// Xavier/Kaiming init tests

#[test]
fn test_exec_xavier_uniform_init_scale() {
    // A param with Xavier uniform should have values in [-a, a]
    // where a = sqrt(6 / (fan_in + fan_out))
    let src = r#"
@model { name: "XavierTest"; }
@graph Forward {
    input x: Tensor<[1, 10], f64>;
    param w: Tensor<[5, 10], f64> { init: "xavier_uniform"; };
    node wt { op: transpose(w); };
    node out { op: matmul(x, wt); };
    output out;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    // fan_in=10, fan_out=5 → a = sqrt(6/(10+5)) = sqrt(0.4) ≈ 0.6325
    let expected_a = (6.0_f64 / 15.0).sqrt();

    let x = CpuTensor::from_f64_slice(&[0.0; 10], (1, 10), DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);
    // Just verify it runs without error (init happens during construction)
    let _ = exec.run("Forward", &inputs).unwrap();

    // Verify the parameter was initialized with correct scale
    let params = exec.graph_params("Forward");
    assert_eq!(params.len(), 1);
    let w_vals = params[0].to_f64_vec().unwrap();
    let max_abs = w_vals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    // Values should be within the Xavier bound (allowing small slack for uniform dist)
    assert!(
        max_abs <= expected_a + 0.01,
        "Xavier uniform max |w| = {}, expected <= {} (a=sqrt(6/15))",
        max_abs,
        expected_a
    );
}

#[test]
fn test_exec_kaiming_normal_init_scale() {
    let src = r#"
@model { name: "KaimingTest"; }
@graph Forward {
    input x: Tensor<[1, 100], f64>;
    param w: Tensor<[50, 100], f64> { init: "kaiming_normal"; };
    node wt { op: transpose(w); };
    node out { op: matmul(x, wt); };
    output out;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);
    let exec = make_executor(src, config);

    // fan_in=100, std = sqrt(2/100) = sqrt(0.02) ≈ 0.1414
    let expected_std = (2.0_f64 / 100.0).sqrt();

    let params = exec.graph_params("Forward");
    assert_eq!(params.len(), 1);
    let w_vals = params[0].to_f64_vec().unwrap();
    let n = w_vals.len() as f64;
    let mean = w_vals.iter().sum::<f64>() / n;
    let var = w_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    // With 5000 elements, empirical std should be close to expected
    assert!(
        (std - expected_std).abs() < 0.05,
        "Kaiming normal std = {}, expected ≈ {} (sqrt(2/100))",
        std,
        expected_std
    );
}
