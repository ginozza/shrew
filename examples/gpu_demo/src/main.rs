//! GPU Demo — Demonstrates Shrew's CUDA backend on a real NVIDIA GPU.
//!
//! Run with: `cargo run --release -p example-gpu-demo`
//!
//! This example shows:
//!   1. Creating tensors on the GPU
//!   2. Element-wise operations (add, mul, relu, gelu, etc.)
//!   3. Matrix multiplication via cuBLAS
//!   4. Reductions (sum, mean, max, min)
//!   5. CPU vs GPU performance comparison
//!   6. Memory pool statistics

use shrew::prelude::*;
use shrew_cuda::{CudaBackend, CudaDevice};
use std::time::Instant;

type GpuTensor = Tensor<CudaBackend>;
type CpuTensor = Tensor<CpuBackend>;

const CPU: CpuDevice = CpuDevice;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║             Shrew — CUDA GPU Backend Demo                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ── 1. Create GPU device ────────────────────────────────────────────
    println!("1. Initializing GPU...");
    let start = Instant::now();
    let gpu = CudaDevice::new(0).expect("Failed to create CUDA device");
    println!("   Device:          {:?}", gpu);
    println!("   Kernel compile:  {:.2?}\n", start.elapsed());

    // ── 2. Basic tensor creation ────────────────────────────────────────
    println!("2. Creating tensors on GPU...");

    let zeros = GpuTensor::zeros((3, 4), DType::F32, &gpu).unwrap();
    println!(
        "   zeros(3,4)  shape={:?}  dtype={:?}",
        zeros.shape(),
        zeros.dtype()
    );

    let ones = GpuTensor::ones((3, 4), DType::F32, &gpu).unwrap();
    println!(
        "   ones(3,4)   shape={:?}  dtype={:?}",
        ones.shape(),
        ones.dtype()
    );

    let randn = GpuTensor::randn((3, 4), DType::F32, &gpu).unwrap();
    println!(
        "   randn(3,4)  shape={:?}  dtype={:?}",
        randn.shape(),
        randn.dtype()
    );

    // Read values back to host
    let vals = randn.to_f64_vec().unwrap();
    println!(
        "   randn data (first 8): {:?}\n",
        &vals[..8.min(vals.len())]
    );

    // ── 3. Element-wise operations ──────────────────────────────────────
    println!("3. Element-wise operations...");

    let a = GpuTensor::randn((1024,), DType::F32, &gpu).unwrap();
    let b = GpuTensor::randn((1024,), DType::F32, &gpu).unwrap();

    let c = a.add(&b).unwrap();
    println!("   add:    shape={:?}", c.shape());

    let d = a.mul(&b).unwrap();
    println!("   mul:    shape={:?}", d.shape());

    let r = a.relu().unwrap();
    println!("   relu:   shape={:?}", r.shape());

    let g = a.gelu().unwrap();
    println!("   gelu:   shape={:?}", g.shape());

    let s = a.sigmoid().unwrap();
    println!("   sigmoid: shape={:?}", s.shape());

    let t = a.tanh().unwrap();
    println!("   tanh:   shape={:?}", t.shape());

    let e = a.exp().unwrap();
    println!("   exp:    shape={:?}", e.shape());

    println!();

    // ── 4. Matrix multiplication (cuBLAS) ───────────────────────────────
    println!("4. Matrix multiplication via cuBLAS...");

    for &sz in &[64usize, 128, 256, 512, 1024] {
        let ma = GpuTensor::randn((sz, sz), DType::F32, &gpu).unwrap();
        let mb = GpuTensor::randn((sz, sz), DType::F32, &gpu).unwrap();

        // Warmup
        let _ = ma.matmul(&mb).unwrap();

        let iters = if sz <= 128 {
            100
        } else if sz <= 256 {
            50
        } else if sz <= 512 {
            20
        } else {
            10
        };
        let start = Instant::now();
        for _ in 0..iters {
            let _ = ma.matmul(&mb).unwrap();
        }
        let per_iter = start.elapsed() / iters;
        let gflops = (2.0 * (sz as f64).powi(3)) / per_iter.as_secs_f64() / 1e9;
        println!("   matmul [{sz:>4}x{sz:>4}] x [{sz:>4}x{sz:>4}]  {per_iter:>10.2?}/iter  {gflops:>8.1} GFLOP/s");
    }
    println!();

    // ── 5. Reductions ───────────────────────────────────────────────────
    println!("5. Reductions...");

    let big = GpuTensor::randn((1000, 1000), DType::F32, &gpu).unwrap();

    let sum_vals = big.sum_all().unwrap().to_f64_vec().unwrap();
    let sum_val = sum_vals[0];
    println!("   sum(1000x1000 randn):  {sum_val:.4}");

    let mean_vals = big.mean_all().unwrap().to_f64_vec().unwrap();
    let mean_val = mean_vals[0];
    println!("   mean(1000x1000 randn): {mean_val:.6}  (expected ~0)");

    println!();

    // ── 6. CPU vs GPU comparison ────────────────────────────────────────
    println!("6. CPU vs GPU performance comparison...\n");
    println!(
        "   {:>30}  {:>12}  {:>12}  {:>8}",
        "Operation", "CPU", "GPU", "Speedup"
    );
    println!("   {}", "-".repeat(68));

    for &sz in &[256usize, 512, 1024] {
        // --- Matmul ---
        let ca = CpuTensor::randn((sz, sz), DType::F32, &CPU).unwrap();
        let cb = CpuTensor::randn((sz, sz), DType::F32, &CPU).unwrap();
        let ga = GpuTensor::randn((sz, sz), DType::F32, &gpu).unwrap();
        let gb = GpuTensor::randn((sz, sz), DType::F32, &gpu).unwrap();

        let iters = if sz <= 256 {
            30
        } else if sz <= 512 {
            10
        } else {
            5
        };

        // warmup
        let _ = ca.matmul(&cb).unwrap();
        let _ = ga.matmul(&gb).unwrap();

        let start = Instant::now();
        for _ in 0..iters {
            let _ = ca.matmul(&cb).unwrap();
        }
        let cpu_time = start.elapsed() / iters;

        let start = Instant::now();
        for _ in 0..iters {
            let _ = ga.matmul(&gb).unwrap();
        }
        let gpu_time = start.elapsed() / iters;

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!(
            "   {:>30}  {:>12.2?}  {:>12.2?}  {:>7.1}x",
            format!("matmul [{sz}x{sz}]"),
            cpu_time,
            gpu_time,
            speedup
        );
    }

    for &sz in &[100_000usize, 1_000_000, 10_000_000] {
        // --- Add ---
        let ca = CpuTensor::randn(vec![sz], DType::F32, &CPU).unwrap();
        let cb = CpuTensor::randn(vec![sz], DType::F32, &CPU).unwrap();
        let ga = GpuTensor::randn(vec![sz], DType::F32, &gpu).unwrap();
        let gb = GpuTensor::randn(vec![sz], DType::F32, &gpu).unwrap();

        let iters = if sz <= 100_000 {
            200
        } else if sz <= 1_000_000 {
            50
        } else {
            10
        };

        let _ = ca.add(&cb).unwrap();
        let _ = ga.add(&gb).unwrap();

        let start = Instant::now();
        for _ in 0..iters {
            let _ = ca.add(&cb).unwrap();
        }
        let cpu_time = start.elapsed() / iters;

        let start = Instant::now();
        for _ in 0..iters {
            let _ = ga.add(&gb).unwrap();
        }
        let gpu_time = start.elapsed() / iters;

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        println!(
            "   {:>30}  {:>12.2?}  {:>12.2?}  {:>7.1}x",
            format!("add [{}]", fmt_size(sz)),
            cpu_time,
            gpu_time,
            speedup
        );
    }

    println!();

    // ── 7. Memory pool stats ────────────────────────────────────────────
    println!("7. Memory pool statistics...");
    let stats = gpu.pool_stats();
    println!(
        "   Cached bytes:   {} ({:.2} MB)",
        stats.cached_bytes,
        stats.cached_bytes as f64 / 1e6
    );
    println!("   Cached buffers: {}", stats.cached_buffers);
    println!("   Pool hits:      {}", stats.hits);
    println!("   Pool misses:    {}", stats.misses);

    println!("\nDone! GPU backend is fully operational. 🚀");
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
