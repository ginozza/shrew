# Architecture: shrew

The `shrew` crate serves as the primary entry point for users. It re-exports functionality from all other crates (`core`, `nn`, `optim`, `data`) to provide a cohesive API. Additionally, it implements high-level features that span multiple domains, such as model serialization and distributed training.

## Core Concepts

- **Facade Pattern**: Unifies the disjointed crate ecosystem into a single `use shrew::prelude::*;`.
- **Serialization**: Supports saving/loading models using standard formats like ONNX and Safetensors, as well as Shrew's native checkpointing.
- **Quantization**: Tools for reducing model size and latency by converting weights to lower precision (INT8, INT4).
- **Distributed**: Primitives for Data Parallelism (DDP) across multiple GPUs or nodes.
- **Profiling**: Instrumentation to analyze performance bottlenecks in the computation graph.

## File Structure

| File | Description | Lines of Code |
| :--- | :--- | :--- |
| `onnx.rs` | Implements export of Shrew computation graphs to the ONNX standard format for interoperability. | 1733 |
| `checkpoint.rs` | Logic for saving and restoring full model states (parameters + optimizer state) to disk. | 936 |
| `profiler.rs` | A performance profiler that tracks operator execution time, memory usage, and shape information. | 901 |
| `distributed.rs` | Implements synchronization primitives (all-reduce) for distributed training strategies. | 841 |
| `quantize.rs` | Algorithms for post-training quantization and quantization-aware training support. | 680 |
| `safetensors.rs` | Integration with the Hugging Face `safetensors` format for secure and fast model weight loading. | 625 |
| `lib.rs` | The root file that re-exports modules and defines the prelude. | 112 |
