# Execution Engine

The execution engine in `shrew-core` takes an optimized graph and runs it.

## Tensors & Storage

Tensors are backed by a storage enum that can hold data on different devices:

```rust
pub enum Storage {
    Cpu(Vec<f32>),
    Cuda(CudaSlice<f32>),
    // ...
}
```

## Backend Dispatch

Operations are dispatched dynamically based on the tensor's device. The `Backend` trait defines the interface for all supported operations (matmul, add, relu, etc.).
