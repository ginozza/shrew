//! Simple performance benchmark for CPU operations.
//!
//! Run with: `cargo run --release -p example-bench-ops`

use shrew_core::dtype::DType;
use shrew_core::tensor::Tensor;
use shrew_cpu::{CpuBackend, CpuDevice};
use std::time::Instant;

type T = Tensor<CpuBackend>;

const DEV: CpuDevice = CpuDevice;

fn bench<F: FnMut()>(name: &str, iters: u32, mut f: F) {
    // Warmup
    for _ in 0..3 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters;
    println!("  {name:<45} {per_iter:>10.2?}  ({iters} iters, {elapsed:.2?} total)");
}

fn main() {
    println!("=== Shrew CPU Performance Benchmark ===\n");

    // Matmul benchmarks (GEMM-accelerated)
    println!(" Matrix Multiplication (gemm) ");

    for &sz in &[64usize, 128, 256, 512] {
        let a = T::randn((sz, sz), DType::F32, &DEV).unwrap();
        let b = T::randn((sz, sz), DType::F32, &DEV).unwrap();
        let iters = if sz <= 128 {
            100
        } else if sz <= 256 {
            20
        } else {
            5
        };
        bench(&format!("matmul [{sz}x{sz}] x [{sz}x{sz}]"), iters, || {
            let _ = a.matmul(&b).unwrap();
        });
    }

    // Batched matmul
    {
        let a = T::randn((8, 64, 64), DType::F32, &DEV).unwrap();
        let b = T::randn((8, 64, 64), DType::F32, &DEV).unwrap();
        bench("batched matmul [8,64,64]", 50, || {
            let _ = a.matmul(&b).unwrap();
        });
    }

    println!();

    // Elementwise binary ops
    println!(" Binary Ops (add, mul) ");

    for &sz in &[1_000usize, 10_000, 100_000, 1_000_000] {
        let a = T::randn(vec![sz], DType::F32, &DEV).unwrap();
        let b = T::randn(vec![sz], DType::F32, &DEV).unwrap();
        let iters = if sz <= 10_000 {
            500
        } else if sz <= 100_000 {
            100
        } else {
            20
        };
        bench(&format!("add [{}]", fmt_size(sz)), iters, || {
            let _ = a.add(&b).unwrap();
        });
    }

    for &sz in &[1_000usize, 100_000, 1_000_000] {
        let a = T::randn(vec![sz], DType::F32, &DEV).unwrap();
        let b = T::randn(vec![sz], DType::F32, &DEV).unwrap();
        let iters = if sz <= 10_000 {
            500
        } else if sz <= 100_000 {
            100
        } else {
            20
        };
        bench(&format!("mul [{}]", fmt_size(sz)), iters, || {
            let _ = a.mul(&b).unwrap();
        });
    }

    println!();

    // Unary ops
    println!(" Unary Ops (exp, relu, gelu) ");

    for &sz in &[10_000usize, 100_000, 1_000_000] {
        let a = T::randn(vec![sz], DType::F32, &DEV).unwrap();
        let iters = if sz <= 10_000 {
            500
        } else if sz <= 100_000 {
            100
        } else {
            20
        };

        bench(&format!("exp [{}]", fmt_size(sz)), iters, || {
            let _ = a.exp().unwrap();
        });
        bench(&format!("relu [{}]", fmt_size(sz)), iters, || {
            let _ = a.relu().unwrap();
        });
        bench(&format!("gelu [{}]", fmt_size(sz)), iters, || {
            let _ = a.gelu().unwrap();
        });
    }

    println!();

    // End-to-end: Linear layer forward
    println!(" Linear Layer Forward ");

    for &(batch, in_f, out_f) in &[(32, 784, 256), (64, 512, 512), (128, 256, 128)] {
        let x = T::randn((batch, in_f), DType::F32, &DEV).unwrap();
        let w = T::randn((out_f, in_f), DType::F32, &DEV).unwrap();
        let bias = T::randn(vec![out_f], DType::F32, &DEV).unwrap();
        let wt = w.t().unwrap();
        bench(
            &format!("linear [{batch},{in_f}]x[{in_f},{out_f}]+bias"),
            20,
            || {
                let out = x.matmul(&wt).unwrap();
                let _ = out.add(&bias).unwrap();
            },
        );
    }

    println!("\nDone.");
}

fn fmt_size(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        format!("{n}")
    }
}
