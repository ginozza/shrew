// JIT Compilation Tests — Verifies compiled graph execution matches interpreter

use std::collections::HashMap;

use shrew::exec::{load_jit, JitExecutor, RuntimeConfig};
use shrew::prelude::*;

// Helper

fn make_jit(src: &str, config: RuntimeConfig) -> JitExecutor<CpuBackend> {
    load_jit::<CpuBackend>(src, CpuDevice, config).expect("failed to JIT compile")
}

fn assert_close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "elem {}: {} vs {} (tol={})",
            i,
            x,
            y,
            tol
        );
    }
}

// Basic graph compilation

#[test]
fn test_jit_identity() {
    let src = r#"
@model { name: "Id"; }
@graph Forward {
    input x: Tensor<[2, 3], f64>;
    node out { op: x; };
    output out;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let x = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = jit.run("Forward", &inputs).unwrap();
    let out = result.get("out").unwrap();
    assert_eq!(out.dims(), &[2, 3]);
    assert_eq!(
        out.to_f64_vec().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn test_jit_add() {
    let src = r#"
@model { name: "Add"; }
@graph Forward {
    input a: Tensor<[3], f64>;
    input b: Tensor<[3], f64>;
    node c { op: a + b; };
    output c;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &CpuDevice).unwrap();
    let b = CpuTensor::from_f64_slice(&[10.0, 20.0, 30.0], 3, DType::F64, &CpuDevice).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), a);
    inputs.insert("b".to_string(), b);

    let result = jit.run("Forward", &inputs).unwrap();
    let c = result.get("c").unwrap();
    assert_eq!(c.to_f64_vec().unwrap(), vec![11.0, 22.0, 33.0]);
}

#[test]
fn test_jit_binary_ops() {
    let src = r#"
@model { name: "BinOps"; }
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
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let x =
        CpuTensor::from_f64_slice(&[10.0, 20.0, 30.0, 40.0], 4, DType::F64, &CpuDevice).unwrap();
    let y = CpuTensor::from_f64_slice(&[2.0, 4.0, 5.0, 8.0], 4, DType::F64, &CpuDevice).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);
    inputs.insert("y".to_string(), y);

    let result = jit.run("Forward", &inputs).unwrap();
    let out = result.get("quot").unwrap();
    assert_eq!(out.to_f64_vec().unwrap(), vec![5.0, 5.0, 6.0, 5.0]);
}

#[test]
fn test_jit_unary_ops() {
    let src = r#"
@model { name: "Unary"; }
@graph Forward {
    input x: Tensor<[3], f64>;
    node r { op: relu(x); };
    output r;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let x = CpuTensor::from_f64_slice(&[-1.0, 0.0, 2.0], 3, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = jit.run("Forward", &inputs).unwrap();
    let out = result.get("r").unwrap();
    assert_eq!(out.to_f64_vec().unwrap(), vec![0.0, 0.0, 2.0]);
}

#[test]
fn test_jit_chain() {
    let src = r#"
@model { name: "Chain"; }
@graph Forward {
    input x: Tensor<[4], f64>;
    input y: Tensor<[4], f64>;
    node sum { op: x + y; };
    node doubled { op: sum + sum; };
    output doubled;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], 4, DType::F64, &CpuDevice).unwrap();
    let y = CpuTensor::from_f64_slice(&[1.0, 1.0, 1.0, 1.0], 4, DType::F64, &CpuDevice).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);
    inputs.insert("y".to_string(), y);

    let result = jit.run("Forward", &inputs).unwrap();
    let out = result.get("doubled").unwrap();
    // (x + y) + (x + y) = [2, 3, 4, 5] + [2, 3, 4, 5] = [4, 6, 8, 10]
    assert_eq!(out.to_f64_vec().unwrap(), vec![4.0, 6.0, 8.0, 10.0]);
}

// Constants

#[test]
fn test_jit_constants() {
    let src = r#"
@model { name: "Const"; }
@graph Forward {
    input x: Tensor<[3], f64>;
    node doubled { op: x + x; };
    output doubled;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = jit.run("Forward", &inputs).unwrap();
    let out = result.get("doubled").unwrap();
    assert_eq!(out.to_f64_vec().unwrap(), vec![2.0, 4.0, 6.0]);
}

// Reductions

#[test]
fn test_jit_sum_all() {
    let src = r#"
@model { name: "Sum"; }
@graph Forward {
    input x: Tensor<[4], f64>;
    node s { op: sum(x); };
    output s;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], 4, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = jit.run("Forward", &inputs).unwrap();
    let out = result.get("s").unwrap();
    let val = out.to_f64_vec().unwrap();
    assert_close(&val, &[10.0], 1e-10);
}

#[test]
fn test_jit_mean_all() {
    let src = r#"
@model { name: "Mean"; }
@graph Forward {
    input x: Tensor<[4], f64>;
    node m { op: mean(x); };
    output m;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let x = CpuTensor::from_f64_slice(&[2.0, 4.0, 6.0, 8.0], 4, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = jit.run("Forward", &inputs).unwrap();
    let out = result.get("m").unwrap();
    let val = out.to_f64_vec().unwrap();
    assert_close(&val, &[5.0], 1e-10);
}

// Matmul

#[test]
fn test_jit_matmul() {
    let src = r#"
@model { name: "MatMul"; }
@graph Forward {
    input a: Tensor<[2, 3], f64>;
    input b: Tensor<[3, 2], f64>;
    node c { op: matmul(a, b); };
    output c;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let a = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();
    let b = CpuTensor::from_f64_slice(
        &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        (3, 2),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), a);
    inputs.insert("b".to_string(), b);

    let result = jit.run("Forward", &inputs).unwrap();
    let c = result.get("c").unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    // [1,2,3] * [[1,0],[0,1],[1,1]] = [4, 5]
    // [4,5,6] * [[1,0],[0,1],[1,1]] = [10, 11]
    assert_close(&c.to_f64_vec().unwrap(), &[4.0, 5.0, 10.0, 11.0], 1e-10);
}

// Parameters and Linear layer

#[test]
fn test_jit_linear_with_params() {
    let src = r#"
@model { name: "LinNet"; }
@graph Forward {
    input x: Tensor<[1, 4], f32>;
    param W: Tensor<[2, 4], f32> { init: "zeros"; };
    param b: Tensor<[2], f32> { init: "zeros"; };
    node out { op: linear(x, W, b); };
    output out;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default());

    let x =
        CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], (1, 4), DType::F32, &CpuDevice).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = jit.run("Forward", &inputs).unwrap();
    let out = result.get("out").unwrap();
    assert_eq!(out.dims(), &[1, 2]);
    // W=zeros, b=zeros => output all zeros
    assert_close(&out.to_f64_vec().unwrap(), &[0.0, 0.0], 1e-10);
}



// Repeated execution (should produce same results)

#[test]
fn test_jit_repeated_runs() {
    let src = r#"
@model { name: "Repeat"; }
@graph Forward {
    input x: Tensor<[3], f64>;
    node y { op: x + x; };
    output y;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    // Run 10 times — should always give same result
    for _ in 0..10 {
        let result = jit.run("Forward", &inputs).unwrap();
        let out = result.get("y").unwrap();
        assert_eq!(out.to_f64_vec().unwrap(), vec![2.0, 4.0, 6.0]);
    }
}

// Dump (debugging)

#[test]
fn test_jit_dump() {
    let src = r#"
@model { name: "Dump"; }
@graph Forward {
    input x: Tensor<[3], f64>;
    node r { op: relu(x); };
    output r;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let dump = jit.dump("Forward").unwrap();
    assert!(dump.contains("JIT Compiled: Forward"));
    assert!(dump.contains("Outputs:"));
    println!("{}", dump);
}

// Activations

#[test]
fn test_jit_sigmoid() {
    let src = r#"
@model { name: "Sig"; }
@graph Forward {
    input x: Tensor<[3], f64>;
    node s { op: sigmoid(x); };
    output s;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let x = CpuTensor::from_f64_slice(&[0.0, 100.0, -100.0], 3, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = jit.run("Forward", &inputs).unwrap();
    let out = result.get("s").unwrap();
    let data = out.to_f64_vec().unwrap();
    assert_close(&[data[0]], &[0.5], 1e-6);
    assert!(data[1] > 0.99);
    assert!(data[2] < 0.01);
}

#[test]
fn test_jit_exp_log() {
    let src = r#"
@model { name: "ExpLog"; }
@graph Forward {
    input x: Tensor<[3], f64>;
    node e { op: exp(x); };
    node l { op: log(e); };
    output l;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], 3, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = jit.run("Forward", &inputs).unwrap();
    let out = result.get("l").unwrap();
    // log(exp(x)) = x
    assert_close(&out.to_f64_vec().unwrap(), &[1.0, 2.0, 3.0], 1e-10);
}

// Reshape

#[test]
fn test_jit_neg() {
    let src = r#"
@model { name: "Neg"; }
@graph Forward {
    input x: Tensor<[4], f64>;
    node n { op: -x; };
    output n;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let x = CpuTensor::from_f64_slice(&[1.0, -2.0, 3.0, -4.0], 4, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = jit.run("Forward", &inputs).unwrap();
    let out = result.get("n").unwrap();
    assert_eq!(out.to_f64_vec().unwrap(), vec![-1.0, 2.0, -3.0, 4.0]);
}

// Softmax

#[test]
fn test_jit_softmax() {
    let src = r#"
@model { name: "Softmax"; }
@graph Forward {
    input x: Tensor<[4], f64>;
    node s { op: softmax(x, dim: 0); };
    output s;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let x = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], 4, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = jit.run("Forward", &inputs).unwrap();
    let out = result.get("s").unwrap();
    let data = out.to_f64_vec().unwrap();
    let sum: f64 = data.iter().sum();
    assert_close(&[sum], &[1.0], 1e-6); // softmax sums to 1
}

// Dead value elimination (free instructions)

#[test]
fn test_jit_dead_value_free() {
    let src = r#"
@model { name: "DeadVal"; }
@graph Forward {
    input x: Tensor<[3], f64>;
    input y: Tensor<[3], f64>;
    node temp { op: x + y; };
    node out { op: temp * x; };
    output out;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let stats = jit.stats("Forward").unwrap();
    // Verify compilation stats are populated
    assert!(stats.num_instructions > 0);
    assert!(stats.num_slots > 0);
}

// Multi-output (only last output)

#[test]
fn test_jit_multi_node_single_output() {
    let src = r#"
@model { name: "Multi"; }
@graph Forward {
    input x: Tensor<[4], f64>;
    node a { op: x + x; };
    node b { op: a + x; };
    node c { op: b + a; };
    output c;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let x = CpuTensor::from_f64_slice(&[1.0, 1.0, 1.0, 1.0], 4, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = jit.run("Forward", &inputs).unwrap();
    let out = result.get("c").unwrap();
    // a = x + x = [2,2,2,2], b = a + x = [3,3,3,3], c = b + a = [5,5,5,5]
    assert_eq!(out.to_f64_vec().unwrap(), vec![5.0, 5.0, 5.0, 5.0]);
}

// Comparison with interpreter

#[test]
fn test_jit_matches_interpreter() {
    let src = r#"
@model { name: "CmpTest"; }
@graph Forward {
    input x: Tensor<[2, 4], f64>;
    input y: Tensor<[2, 4], f64>;
    node sum { op: x + y; };
    node sq { op: sum * sum; };
    node neg { op: -sq; };
    node act { op: relu(neg); };
    output act;
}
"#;
    let config = RuntimeConfig::default().with_dtype(DType::F64);

    // Build with interpreter
    use shrew::exec::load_program;
    let interp = load_program::<CpuBackend>(src, CpuDevice, config.clone())
        .expect("interpreter load failed");

    // Build with JIT
    let jit = make_jit(src, config);

    let x = CpuTensor::from_f64_slice(
        &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0],
        (2, 4),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();
    let y = CpuTensor::from_f64_slice(
        &[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        (2, 4),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);
    inputs.insert("y".to_string(), y);

    let interp_result = interp.run("Forward", &inputs).unwrap();
    let jit_result = jit.run("Forward", &inputs).unwrap();

    let interp_out = interp_result.get("act").unwrap().to_f64_vec().unwrap();
    let jit_out = jit_result.get("act").unwrap().to_f64_vec().unwrap();

    assert_close(&interp_out, &jit_out, 1e-12);
}

// Transpose

#[test]
fn test_jit_transpose() {
    let src = r#"
@model { name: "Trans"; }
@graph Forward {
    input x: Tensor<[2, 3], f64>;
    node t { op: transpose(x); };
    output t;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let x = CpuTensor::from_f64_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        DType::F64,
        &CpuDevice,
    )
    .unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = jit.run("Forward", &inputs).unwrap();
    let out = result.get("t").unwrap();
    assert_eq!(out.dims(), &[3, 2]);
}

// Comparison ops

#[test]
fn test_jit_comparison() {
    let src = r#"
@model { name: "Cmp"; }
@graph Forward {
    input a: Tensor<[4], f64>;
    input b: Tensor<[4], f64>;
    node eq { op: a == b; };
    output eq;
}
"#;
    let jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0, 4.0], 4, DType::F64, &CpuDevice).unwrap();
    let b = CpuTensor::from_f64_slice(&[1.0, 0.0, 3.0, 5.0], 4, DType::F64, &CpuDevice).unwrap();

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), a);
    inputs.insert("b".to_string(), b);

    let result = jit.run("Forward", &inputs).unwrap();
    let out = result.get("eq").unwrap();
    let data = out.to_f64_vec().unwrap();
    assert_eq!(data[0], 1.0); // 1 == 1
    assert_eq!(data[1], 0.0); // 2 != 0
    assert_eq!(data[2], 1.0); // 3 == 3
    assert_eq!(data[3], 0.0); // 4 != 5
}

// Recompile

#[test]
fn test_jit_recompile() {
    let src = r#"
@model { name: "Recomp"; }
@graph Forward {
    input x: Tensor<[3], f64>;
    node y { op: relu(x); };
    output y;
}
"#;
    let mut jit = make_jit(src, RuntimeConfig::default().with_dtype(DType::F64));

    // Run once
    let x = CpuTensor::from_f64_slice(&[-1.0, 0.0, 1.0], 3, DType::F64, &CpuDevice).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), x);

    let result = jit.run("Forward", &inputs).unwrap();
    assert_eq!(
        result.get("y").unwrap().to_f64_vec().unwrap(),
        vec![0.0, 0.0, 1.0]
    );

    // Recompile and run again
    jit.recompile("Forward").unwrap();
    let result2 = jit.run("Forward", &inputs).unwrap();
    assert_eq!(
        result2.get("y").unwrap().to_f64_vec().unwrap(),
        vec![0.0, 0.0, 1.0]
    );
}
