//! # Shrew
//!
//! A deep learning library built from scratch in Rust.
//!
//! This is the top-level facade crate that re-exports everything you need.
//!
//! ## Usage
//!
//! ```rust
//! use shrew::prelude::*;
//! ```
//!
//! ## Architecture
//!
//! | Crate | Purpose |
//! |-------|----------|
//! | `shrew-core` | Tensor, Shape, DType, Layout, Backend trait, Autograd |
//! | `shrew-cpu` | CPU backend with SIMD matmul and rayon parallelism |
//! | `shrew-nn` | Neural network layers (Linear, Conv2d, RNN, LSTM, Transformer, etc.) |
//! | `shrew-optim` | Optimizers (SGD, Adam, AdamW, RAdam, RMSProp), LR schedulers, EMA |
//! | `shrew-data` | Dataset, DataLoader, MNIST, transforms |
//! | `shrew-cuda` | CUDA GPU backend (feature-gated) |
//! | `shrew-ir` | `.sw` IR format: lexer, parser, AST, Graph IR |
//!
//! ## Modules
//!
//! - [`distributed`] — DataParallel, MixedPrecisionTrainer, PipelineParallel
//! - [`quantize`] — INT8/INT4 post-training quantization
//! - [`onnx`] — ONNX import/export
//! - [`profiler`] — Timing, memory tracking, model benchmarks
//! - [`exec`] — Graph executor for `.sw` programs
//! - [`checkpoint`] — Save/load model parameters
//! - [`safetensors`] — HuggingFace-compatible serialization

/// Re-export core types.
pub use shrew_core::{
    backend::{Backend, BackendDevice, BackendStorage, BinaryOp, CmpOp, ReduceOp, UnaryOp},
    op::{Op, TensorId},
    DType, Error, GradStore, Layout, Result, Shape, Tensor, WithDType,
};

/// Re-export CPU backend.
pub use shrew_cpu::{CpuBackend, CpuDevice, CpuStorage, CpuTensor};

/// Re-export CUDA backend (requires `cuda` feature + NVIDIA CUDA Toolkit).
#[cfg(feature = "cuda")]
pub use shrew_cuda::{CudaBackend, CudaDevice, CudaStorage, CudaTensor};

/// Re-export neural network modules.
pub mod nn {
    pub use shrew_nn::*;
}

/// Re-export optimizers.
pub mod optim {
    pub use shrew_optim::*;
}

/// Re-export the .sw IR parser and AST.
pub mod ir {
    pub use shrew_ir::*;
}

/// Graph executor — runs .sw programs on the tensor runtime.
pub mod exec;

/// Checkpoint — save and load model parameters.
pub mod checkpoint;

/// Safetensors — interoperable tensor serialization (HuggingFace format).
pub mod safetensors;

/// Distributed training — DataParallel, MixedPrecision, Pipeline, gradient sync.
pub mod distributed;

/// Quantization — INT8/INT4 post-training quantization for inference.
pub mod quantize;

/// ONNX — Import/Export for interoperability with other frameworks.
pub mod onnx;

/// Profiling & Benchmarking — op-level timing, memory tracking, model summaries.
pub mod profiler;

/// Prelude: import this for the most common types.
pub mod prelude {
    pub use crate::checkpoint::TrainingCheckpoint;
    pub use crate::distributed::{
        reduce_gradients, AllReduceOp, DataParallel, LossScaleConfig, MixedPrecisionTrainer,
        ParallelTrainer, PipelineParallel, PipelineStage,
    };
    pub use crate::exec::{CompileStats, JitExecutor, JitResult};
    pub use crate::exec::{Executor, RuntimeConfig, Trainer};
    pub use crate::nn::{
        bce_loss, bce_with_logits_loss, cross_entropy_loss, l1_loss, mse_loss, nll_loss,
        smooth_l1_loss,
    };
    pub use crate::nn::{
        AdaptiveAvgPool2d, AvgPool2d, BatchNorm2d, Conv1d, Conv2d, Dropout, Flatten, GRUCell, GeLU,
        GroupNorm, LSTMCell, LayerNorm, LeakyReLU, Linear, MaxPool2d, Mish, Module,
        MultiHeadAttention, RMSNorm, RNNCell, ReLU, Sequential, SiLU, TransformerBlock, ELU, GRU,
        LSTM, RNN,
    };
    pub use crate::onnx::{
        export_tensors as export_onnx_tensors, export_weights as export_onnx, load_onnx_weights,
        OnnxAttribute, OnnxModel, OnnxNode, OnnxTensor,
    };
    pub use crate::optim::EMA;
    pub use crate::optim::{clip_grad_norm, clip_grad_value, grad_norm, GradAccumulator};
    pub use crate::optim::{Adam, AdamW, Optimizer, OptimizerState, RAdam, RMSProp, Stateful, SGD};
    pub use crate::optim::{
        CosineAnnealingLR, CosineWarmupLR, ExponentialLR, LinearLR, LrScheduler, ReduceLROnPlateau,
        StepLR,
    };
    pub use crate::profiler::{
        benchmark_forward, benchmark_forward_backward, estimate_model_memory, format_bytes,
        BenchmarkResult, MemoryTracker, ModelSummary, ProfileEntry, ProfileEvent, ProfileReport,
        Profiler, ScopedTimer, Stopwatch,
    };
    pub use crate::quantize::{
        dequantize_tensor, quantization_stats, quantize_named_parameters, quantize_tensor,
        QuantBits, QuantConfig, QuantGranularity, QuantMode, QuantStats, QuantizedLinear,
        QuantizedTensor,
    };
    pub use crate::{CpuBackend, CpuDevice, CpuTensor, DType, GradStore, Shape, Tensor};
}
