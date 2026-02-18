# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-15

### Added

#### Tensor System (`shrew-core`)
- Multi-dimensional tensor with 7 dtypes: `F16`, `BF16`, `F32`, `F64`, `U8`, `U32`, `I64`
- Shape, Layout, strides, and contiguity tracking
- Constructors: `zeros`, `ones`, `full`, `rand`, `randn`, `from_f64_slice`
- Binary ops: `add`, `sub`, `mul`, `div` with NumPy-style broadcasting
- Unary ops: `neg`, `abs`, `exp`, `log`, `sqrt`, `square`, `sin`, `cos`
- Activations: `relu`, `sigmoid`, `tanh`, `gelu`, `silu`
- Reductions: `sum`, `sum_all`, `mean`, `mean_all`, `max`, `min`, `argmax`, `argmin`
- Shape ops: `reshape`, `transpose`, `t`, `narrow`, `unsqueeze`, `squeeze_all`, `expand`, `contiguous`
- Composite ops: `softmax`, `log_softmax`, `var`, `matmul`, `affine`, `cat`, `chunk`, `index_select`
- Comparison ops: `eq`, `ne`, `gt`, `ge`, `lt`, `le`
- Dtype casting: `to_dtype` with automatic F16/BF16 promote/demote
- Eager reverse-mode automatic differentiation with `backward()`
- Dynamic symbolic shapes: `SymDim`, `SymbolicShape`, `ShapeEnv`, `ShapeGuard`

#### CPU Backend (`shrew-cpu`)
- Full op coverage with `gemm` SIMD-accelerated matmul (AVX2/AVX-512/FMA)
- Parallel batched matmul and large elementwise ops via `rayon`
- Contiguous fast-path bypassing stride calculations
- Broadcasting support for all binary operations

#### CUDA GPU Backend (`shrew-cuda`)
- NVIDIA GPU backend via `cudarc` (cuBLAS + custom PTX kernels)
- Elementwise, reduction, cast, and broadcast kernels
- Memory pool with hit/miss tracking
- Mixed-precision training support (F16/BF16 ↔ F32 casting)

#### Neural Network Layers (`shrew-nn`)
- `Linear`, `Conv1d`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`
- `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `RMSNorm`
- `Embedding`, `Flatten`, `Dropout`, `Sequential`
- `RNNCell`, `RNN`, `LSTMCell`, `LSTM`, `GRUCell`, `GRU`
- `MultiHeadAttention`, `TransformerBlock`
- Activation modules: `ReLU`, `GeLU`, `SiLU`, `LeakyReLU`, `ELU`, `Mish`
- Loss functions: `mse_loss`, `cross_entropy_loss`, `l1_loss`, `smooth_l1_loss`, `bce_loss`, `bce_with_logits_loss`, `nll_loss`

#### Optimizers (`shrew-optim`)
- `SGD` with momentum and weight decay
- `Adam`, `AdamW`, `RAdam`, `RMSProp`
- LR schedulers: `StepLR`, `ExponentialLR`, `LinearLR`, `CosineAnnealingLR`, `CosineWarmupLR`, `ReduceLROnPlateau`
- Gradient utilities: `clip_grad_norm`, `clip_grad_value`, `grad_norm`, `GradAccumulator`
- `EMA` (Exponential Moving Average)

#### IR & Compiler (`shrew-ir`)
- `.sw` declarative model format: lexer, parser, AST
- Graph IR with lowering, validation, shape inference
- Optimization passes: DCE, CSE, constant folding, operator fusion, identity elimination
- Computed dimensions and symbolic shape support

#### Executor & Training (`shrew`)
- Graph interpreter from IR
- JIT graph compilation: `CompiledGraph`, `MemoryPlan`, `JitExecutor`, `load_jit`
- `Trainer` with training loop, validation, early stopping, metric tracking
- Distributed training: `DataParallel`, `PipelineParallel`, `MixedPrecisionTrainer`
- INT8/INT4 post-training quantization (`QuantizedLinear`)
- ONNX export/import (opset 17, built-in protobuf, zero dependencies)
- Profiling: `Profiler`, `MemoryTracker`, `ModelSummary`, `benchmark_forward`
- Serialization: `.shrew` binary checkpoints, HuggingFace safetensors

#### Data Loading (`shrew-data`)
- `Dataset` trait, `DataLoader` with shuffling and batching
- Built-in MNIST dataset with automatic download
- Image transforms: resize, normalize, random crop, flip, rotation, color jitter, random erasing
- Async prefetch loader

#### Python Bindings (`shrew-python`)
- PyO3 bindings for tensors, modules, optimizers, data loading
- NumPy interop (`from_numpy`, `to_numpy`)
- `.sw` file execution from Python

#### CLI (`shrew-cli`)
- `shrew dump <file.sw>` — print lowered IR graph
- `shrew validate <file.sw>` — check program for errors
- `shrew bench <file.sw>` — benchmark forward pass
- `shrew info <file.sw>` — show model summary

#### Examples
- `linear_regression` — autograd demo
- `mlp_xor` — MLP trained with Adam on XOR
- `mnist` — MNIST digit classification (MLP + DataLoader)
- `mnist_cnn` — MNIST CNN (Conv2d + BatchNorm + MaxPool)
- `char_gpt` — character-level GPT transformer
- `bench_ops` — CPU/GPU performance benchmarks
- `rnn_sequence` — RNN/LSTM/GRU sequence modeling
- `gpu_demo` — CUDA backend demo with CPU vs GPU comparison

[0.1.0]: https://github.com/ginozza/shrew/releases/tag/v0.1.0
