<p align="center">
  <img src="assets/shrew_banner.svg" alt="Shrew" width="100%">
</p>

<p align="center">
  <a href="https://github.com/ginozza/shrew/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/ginozza/shrew/ci.yml?branch=master&label=CI" alt="CI"></a>
  <a href="https://github.com/ginozza/shrew/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <a href="https://crates.io/crates/shrew"><img src="https://img.shields.io/crates/v/shrew.svg" alt="Crates.io"></a>
  <img src="https://img.shields.io/badge/rust-1.75%2B-orange.svg" alt="MSRV">
</p>

---

Shrew is a modular deep learning framework written from scratch in Rust. It provides a tensor system with automatic differentiation, neural network layers, optimizers, a CUDA GPU backend, and a declarative intermediate representation (`.sw`) for language-agnostic model specification.

Models defined in `.sw` files can be trained from Python (via PyO3 bindings) or Rust, and deployed with zero transpilation across platforms.

## Components

| Crate | Description |
|-------|-------------|
| **shrew-core** | `Tensor<B>`, `Shape`, `DType`, `Layout`, `Backend` trait, reverse-mode autograd, dynamic symbolic shapes |
| **shrew-cpu** | CPU backend: SIMD matmul via `gemm` (AVX2/AVX-512/FMA), parallel ops via `rayon`, broadcasting |
| **shrew-cuda** | NVIDIA GPU backend: cuBLAS matmul, custom PTX kernels, memory pool, mixed-precision F16/BF16 |
| **shrew-nn** | Neural network layers: Linear, Conv1d/2d, RNN/LSTM/GRU, MultiHeadAttention, Transformer, BatchNorm, LayerNorm, losses |
| **shrew-optim** | Optimizers (SGD, Adam, AdamW, RAdam, RMSProp), LR schedulers, gradient clipping, EMA |
| **shrew-ir** | `.sw` format: lexer, parser, AST, Graph IR, lowering, validation, shape inference, optimization passes |
| **shrew-data** | `Dataset` trait, `DataLoader`, MNIST, image transforms, async prefetch loader |
| **shrew** | Facade crate: executor, JIT compiler, trainer, distributed training, quantization, ONNX, profiling, checkpoints |
| **shrew-python** | Python bindings via PyO3 with NumPy interop |
| **shrew-cli** | CLI tools: `shrew dump`, `validate`, `bench`, `info` |

## Technical Features

### Backend-Agnostic Tensor System

`Tensor<B>` is generic over `Backend`. The same tensor code runs on CPU and GPU without changes. Supported dtypes: `F16`, `BF16`, `F32`, `F64`, `U8`, `U32`, `I64`.

```rust
use shrew::prelude::*;

let dev = CpuDevice;
let a = CpuTensor::randn((3, 4), DType::F32, &dev)?;
let b = CpuTensor::randn((4, 5), DType::F32, &dev)?;
let c = a.matmul(&b)?;  // [3,4] × [4,5] → [3,5]
```

Operations: `add`, `sub`, `mul`, `div` (with NumPy-style broadcasting), `neg`, `abs`, `exp`, `log`, `sqrt`, `square`, `sin`, `cos`, `relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `softmax`, `log_softmax`, `matmul`, `reshape`, `transpose`, `narrow`, `unsqueeze`, `expand`, `cat`, `chunk`, `index_select`, `sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `var`, comparisons (`eq`, `ne`, `gt`, `ge`, `lt`, `le`), `to_dtype`.

### Reverse-Mode Automatic Differentiation

Eager autograd — every op records its computational graph. `backward()` does topological sort and applies the chain rule. Gradient paths cover all binary/unary ops, reductions, matmul, reshape, transpose, narrow, affine, contiguous, cat, and index_select.

```rust
let w = CpuTensor::randn((3, 3), DType::F64, &dev)?.set_variable();
let x = CpuTensor::randn((2, 3), DType::F64, &dev)?;
let loss = x.matmul(&w)?.sum_all()?;
let grads = loss.backward()?;
let dw = grads.get(&w).unwrap();  // ∂loss/∂w
```

### Neural Network Layers

All layers implement `Module::forward()` and are generic over `Backend`:

| Category | Layers |
|----------|--------|
| Dense | `Linear` |
| Convolution | `Conv1d`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d` |
| Recurrent | `RNNCell`, `RNN`, `LSTMCell`, `LSTM`, `GRUCell`, `GRU` |
| Attention | `MultiHeadAttention`, `TransformerBlock` |
| Normalization | `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `RMSNorm` |
| Embedding | `Embedding` |
| Regularization | `Dropout`, `Flatten`, `Sequential` |
| Activations | `ReLU`, `GeLU`, `SiLU`, `LeakyReLU`, `ELU`, `Mish` |
| Losses | `mse_loss`, `cross_entropy_loss`, `l1_loss`, `smooth_l1_loss`, `bce_loss`, `bce_with_logits_loss`, `nll_loss` |

### Optimizers and Schedulers

| Optimizers | Schedulers |
|------------|-----------|
| `SGD` (momentum, weight decay) | `StepLR`, `ExponentialLR`, `LinearLR` |
| `Adam`, `AdamW`, `RAdam` | `CosineAnnealingLR`, `CosineWarmupLR` |
| `RMSProp` | `ReduceLROnPlateau` |

Utilities: `clip_grad_norm`, `clip_grad_value`, `grad_norm`, `GradAccumulator`, `EMA`.

### CUDA GPU Backend

Feature-gated backend using `cudarc`. cuBLAS for matrix multiplication, custom PTX kernels for elementwise, reduction, broadcast, and cast operations. Includes a memory pool with allocation reuse.

```bash
cargo build -p shrew --features cuda
```

Mixed-precision training: `MixedPrecisionTrainer` with dynamic loss scaling, automatic F32↔F16/BF16 casting via `to_dtype`.

### `.sw` Intermediate Representation

Declarative, text-based model specification — separates model architecture from runtime execution:

```sw
@model { name: "TinyGPT"; }

@config {
    d_model: 256;
    n_heads: 4;
    d_ff: 256 * 4;   // constant folding → 1024
}

@graph Forward {
    input tokens: Tensor<[Batch, SeqLen], i64>;
    param wte: Tensor<[50257, 256], f32> { init: "normal(0, 0.02)"; };
    param wpe: Tensor<[512, 256], f32>   { init: "normal(0, 0.02)"; };

    node tok_emb  { op: embedding(wte, tokens); };
    node pos_emb  { op: embedding(wpe, positions); };
    node h        { op: tok_emb + pos_emb; };
    node tf_out   { op: repeat(4) { transformer_block(h, n_heads: 4); }; };
    node ln_out   { op: layer_norm(tf_out, ln_w, ln_b, eps: 1e-5); };
    node logits   { op: matmul(ln_out, transpose(wte)); };
    output logits;
}

@training {
    loss: cross_entropy;
    optimizer: { type: "AdamW"; lr: 3e-4; weight_decay: 0.1; }
    epochs: 20;
    batch_size: 64;
}
```

Pipeline: source → **Lexer** → tokens → **Parser** → AST → **Lowering** → Graph IR → **Validate** → **Shape inference** → **Optimize** (DCE, CSE, constant folding, operator fusion, identity elimination).

### JIT Compilation

`JitExecutor` compiles IR graphs into a flat instruction tape with pre-allocated memory slots and value lifetime tracking. No re-interpretation of the graph at runtime.

```rust
use shrew::exec::jit::load_jit;

let executor = load_jit::<CpuBackend>(sw_source, CpuDevice, config)?;
let result = executor.run("Forward", &inputs)?;
```

### Dynamic Symbolic Shapes

`SymDim` (Fixed/Symbolic/Dynamic), `SymbolicShape`, `ShapeEnv`, and `ShapeGuard` bridge symbolic IR shapes with runtime concrete shapes. Supports shape unification, matching, and broadcasting.

### Distributed Training

| Component | Description |
|-----------|-------------|
| `DataParallel` | Batch splitting across workers with output concatenation |
| `PipelineParallel` | GPipe-style micro-batch pipelining |
| `MixedPrecisionTrainer` | Dynamic loss scaling for FP16/BF16 |
| `reduce_gradients` | All-reduce gradient synchronization |

### Quantization

INT8/INT4 post-training quantization (symmetric/asymmetric, per-tensor/per-channel). `QuantizedLinear` for dequantize-on-the-fly inference.

### ONNX Interop

Export/import ONNX models (opset 17) with a built-in minimal protobuf encoder/decoder (zero external dependencies).

### Profiling

`Profiler` with named timing events, `MemoryTracker`, `ModelSummary`, `benchmark_forward`, `benchmark_forward_backward`.

### Serialization

| Format | Description |
|--------|-------------|
| `.shrew` | Native binary checkpoint (`save_tensors` / `load_tensors`) |
| Safetensors | HuggingFace-compatible (`save_safetensors` / `load_safetensors`) |
| ONNX | Open Neural Network Exchange (`export_weights` / `load_onnx_weights`) |

## Installation

### Rust

```toml
# Cargo.toml
[dependencies]
shrew = "0.1"
```

With CUDA support:

```toml
[dependencies]
shrew = { version = "0.1", features = ["cuda"] }
```

Or directly from GitHub:

```toml
[dependencies]
shrew = { git = "https://github.com/ginozza/shrew" }
```

### CLI

```bash
cargo install shrew-cli
```

Or from source:

```bash
cargo install --git https://github.com/ginozza/shrew shrew-cli
```

This installs the `shrew` binary with commands: `dump`, `validate`, `bench`, `info`.

### Python

```bash
pip install shrew-python
```

Or build from source:

```bash
git clone https://github.com/ginozza/shrew
cd shrew
pip install maturin
maturin develop --release
```

```python
import shrew_python as shrew

t = shrew.tensor([1.0, 2.0, 3.0])
print(t)
```

### From Source (full workspace)

```bash
git clone https://github.com/ginozza/shrew
cd shrew
cargo build --workspace
cargo test --workspace
```

Requirements:
- Rust 1.75+ (edition 2021)
- Python 3.9+ (for Python bindings)
- NVIDIA CUDA Toolkit (for GPU backend only)

## Getting Started

### Tensor operations and autograd

```rust
use shrew::prelude::*;

fn main() -> shrew::Result<()> {
    let dev = CpuDevice;

    // Broadcasting: [3,1] + [1,2] → [3,2]
    let a = CpuTensor::from_f64_slice(&[1.0, 2.0, 3.0], (3, 1), DType::F64, &dev)?;
    let b = CpuTensor::from_f64_slice(&[10.0, 20.0], (1, 2), DType::F64, &dev)?;
    let c = a.add(&b)?;

    // Transformer forward pass
    let block = TransformerBlock::<CpuBackend>::new(64, 4, 256, true, DType::F64, &dev)?;
    let x = CpuTensor::rand((2, 10, 64), DType::F64, &dev)?;
    let y = block.forward(&x)?;  // [2,10,64]

    // Autograd
    let w = CpuTensor::rand((3, 3), DType::F64, &dev)?.set_variable();
    let input = CpuTensor::rand((2, 3), DType::F64, &dev)?;
    let loss = input.matmul(&w)?.sum_all()?;
    let grads = loss.backward()?;

    Ok(())
}
```

### Executing a `.sw` model

```rust
use shrew::prelude::*;
use shrew::exec::{load_program, RuntimeConfig};

let src = r#"
@model { name: "MLP"; }
@graph Forward {
    input x: Tensor<[2, 4], f32>;
    param w: Tensor<[4, 3], f32> { init: "normal(0, 0.1)"; };
    node out { op: softmax(matmul(x, w)); };
    output out;
}
"#;

let config = RuntimeConfig::default().with_dtype(DType::F32);
let exec = load_program::<CpuBackend>(src, CpuDevice, config)?;

let x = CpuTensor::rand((2, 4), DType::F32, &CpuDevice)?;
let mut inputs = std::collections::HashMap::new();
inputs.insert("x".to_string(), x);

let result = exec.run("Forward", &inputs)?;
let probs = result.get("out").unwrap();
assert_eq!(probs.dims(), &[2, 3]);
```

### Quantization

```rust
use shrew::prelude::*;

let model = Linear::<CpuBackend>::new(256, 128, true, DType::F32, &CpuDevice)?;
let config = QuantConfig::int8_per_channel();
let quantized = quantize_named_parameters::<CpuBackend>(&model, &config)?;
```

### Benchmarking

```rust
use shrew::prelude::*;

let model = Linear::<CpuBackend>::new(512, 256, true, DType::F32, &CpuDevice)?;
let result = benchmark_forward(
    &model,
    || Tensor::<CpuBackend>::rand((32, 512), DType::F32, &CpuDevice).unwrap(),
    32, 5, 100,
)?;
println!("{}", result);
```

## Build & Test

```bash
cargo build --workspace           # Build all crates
cargo test --workspace            # Run all tests (~600+)
cargo clippy --workspace          # Lint
cargo fmt --all --check           # Format check
cargo doc --workspace --no-deps   # Generate documentation
```

### Examples

```bash
cargo run -p example-linear-regression
cargo run -p example-mlp-xor
cargo run -p mnist-example                    # Requires MNIST data download
cargo run -p mnist-cnn-example
cargo run --release -p char-gpt-example       # Char-level GPT on Shakespeare
cargo run -p example-rnn-sequence
cargo run --release -p example-bench-ops      # CPU performance benchmarks
```

### CPU Performance (release mode)

| Operation | Size | Time |
|-----------|------|------|
| matmul | 256×256 × 256×256 | ~370 µs |
| matmul | 512×512 × 512×512 | ~3.5 ms |
| add | 1M elements | ~1.3 ms |
| linear forward | [64,512]×[512,512]+bias | ~3.9 ms |

## Dependencies

| Crate | Purpose |
|-------|---------|
| `gemm` | SIMD-accelerated matmul (auto AVX2/AVX-512/FMA) |
| `rayon` | Parallel iteration for batched ops |
| `half` | F16/BF16 with num-traits |
| `cudarc` | CUDA driver/runtime, cuBLAS (optional) |
| `pyo3` / `numpy` | Python bindings (optional) |
| `num-traits` | Numeric trait bounds |
| `rand` / `rand_distr` | Random initialization |
| `thiserror` | Error types |
| `serde_json` | Checkpoint metadata |

## Contributing

Bug fixes are welcome without prior discussion. For new features or architectural changes, please open an issue first. See the [CHANGELOG](CHANGELOG.md) for release history.

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
