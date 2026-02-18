# Crate Structure

The project is organized as a Cargo workspace with the following members:

- **`shrew`**: The top-level crate that re-exports functionality and provides the main entry point (CLI via `shrew-cli`).
- **`shrew-core`**: The runtime core. Handles tensor storage, device management, and graph execution APIs.
- **`shrew-ir`**: Contains the parser, AST definitions, and the Intermediate Representation (IR) logic.
- **`shrew-optim`**: Optimization passes for the IR (graph rewriting, fusion).
- **`shrew-nn`**: Implementation of neural network layers and common operators.
- **`shrew-data`**: Data loading and preprocessing utilities.
- **`shrew-cpu`**: CPU backend implementation (using Rayon and SIMD where available).
- **`shrew-cuda`**: CUDA backend implementation (interfaces with cuBLAS, cuDNN).
- **`shrew-cli`**: Command-line interface tool.
- **`shrew-python`**: Python bindings (PyO3) for using Shrew from Python.
