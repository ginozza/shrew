//! # shrew-core
//!
//! Core tensor primitives, types, backend traits, and autograd for Shrew.
//!
//! This crate provides:
//! - [`Tensor`] — n-dimensional array with automatic differentiation
//! - [`Shape`] / [`Layout`] — shape, strides, and memory layout
//! - [`DType`] — data types (F16, BF16, F32, F64, U8, U32, I64)
//! - [`Backend`] trait — abstraction over CPU/GPU execution
//! - [`GradStore`] — gradient storage returned by `backward()`
// - DType: supported numeric types (f32, f64, etc.)
// - Shape: n-dimensional shape representation
// - Layout: shape + strides + offset for memory layout
// - Backend trait: abstraction for compute backends (CPU, CUDA, etc.)
// - Tensor: the fundamental n-dimensional array type
// - Op/Backprop: computational graph for automatic differentiation (Phase 2)

pub mod backend;
pub mod backprop;
pub mod dtype;
pub mod dynamic_shape;
pub mod error;
pub mod layout;
pub mod op;
pub mod shape;
pub mod tensor;

pub use backend::{Backend, BackendDevice};
pub use backprop::GradStore;
pub use backprop::{checkpoint, checkpoint_sequential, is_checkpointing, with_checkpoint_mode};
pub use dtype::{DType, WithDType};
pub use dynamic_shape::{ShapeEnv, ShapeGuard, SymDim, SymbolicShape};
pub use error::{Error, Result};
pub use layout::Layout;
pub use op::Op;
pub use shape::Shape;
pub use tensor::Tensor;
