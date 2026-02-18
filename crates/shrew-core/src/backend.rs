use crate::dtype::DType;
use crate::error::Result;
use crate::layout::Layout;
use crate::shape::Shape;
use std::fmt;

// Backend — Abstraction over compute devices (CPU, CUDA, etc.)
//
// The Backend trait is the central abstraction that makes Shrew extensible.
// Each backend (CPU, CUDA, ROCm, etc.) implements this trait, providing
// its own storage type and operation implementations.
//
// WHY A TRAIT AND NOT AN ENUM?
//
// Using a trait (vs. an enum like `Device::Cpu | Device::Cuda`) means:
// - New backends can be added as separate crates without modifying shrew-core
// - Each backend can have different associated types (e.g., CudaStorage vs CpuStorage)
// - The compiler can monomorphize for performance
//
// The tradeoff is that Tensor becomes generic: Tensor<B: Backend>.
// This is similar to Burn's approach and provides maximum flexibility.

/// Identifies a compute device (e.g., "CPU", "CUDA:0", "CUDA:1").
pub trait BackendDevice: Clone + fmt::Debug + Send + Sync + 'static {
    /// A human-readable name for this device (e.g., "cpu", "cuda:0").
    fn name(&self) -> String;
}

/// A storage buffer that holds tensor data on a specific device.
///
/// For CPU, this is a `Vec<f32>` (or enum over dtypes).
/// For CUDA, this is a device memory allocation (`CudaSlice`).
pub trait BackendStorage: Clone + Send + Sync + 'static {
    /// The data type of the elements in this storage.
    fn dtype(&self) -> DType;

    /// Total number of elements that fit in this storage.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// Binary and Unary operation enums
//
// These enums serve two purposes:
// 1. They parameterize the backend ops (so we have one trait method per category)
// 2. They are recorded in the Op enum for autograd (knowing WHICH binary op
//    was performed is needed to compute the correct gradient)

/// Element-wise binary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Element-wise unary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    Silu,
    Sin,
    Cos,
    Square,
    Floor,
    Ceil,
    Round,
}

/// Reduction operations along dimension(s).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    ArgMax,
    ArgMin,
}

/// Comparison operations (produce boolean tensors).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

// Backend Trait — The core interface every backend must implement

/// The main Backend trait. Implementing this for a struct (e.g., CpuBackend)
/// makes that struct a complete compute backend for Shrew.
///
/// All operations take storage + layout (which encodes shape/strides) and
/// return new storage (immutable semantics — no in-place mutation by default).
pub trait Backend: Clone + Send + Sync + fmt::Debug + 'static {
    /// The device type for this backend.
    type Device: BackendDevice;
    /// The storage type for this backend.
    type Storage: BackendStorage;

    //  Creation 

    /// Allocate storage filled with zeros.
    fn zeros(shape: &Shape, dtype: DType, device: &Self::Device) -> Result<Self::Storage>;

    /// Allocate storage filled with ones.
    fn ones(shape: &Shape, dtype: DType, device: &Self::Device) -> Result<Self::Storage>;

    /// Allocate storage filled with a constant value.
    fn full(shape: &Shape, val: f64, dtype: DType, device: &Self::Device) -> Result<Self::Storage>;

    /// Create storage from a flat f64 slice, converting to the target dtype.
    fn from_f64_slice(data: &[f64], dtype: DType, device: &Self::Device) -> Result<Self::Storage>;

    /// Create storage with random uniform values in [0, 1).
    fn rand_uniform(shape: &Shape, dtype: DType, device: &Self::Device) -> Result<Self::Storage>;

    /// Create storage with random normal values (mean=0, std=1).
    fn rand_normal(shape: &Shape, dtype: DType, device: &Self::Device) -> Result<Self::Storage>;

    //  Element-wise binary ops 

    /// Apply a binary op element-wise: result[i] = op(lhs[i], rhs[i]).
    /// The layouts handle broadcasting and non-contiguous access.
    fn binary_op(
        op: BinaryOp,
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
    ) -> Result<Self::Storage>;

    //  Element-wise unary ops 

    /// Apply a unary op element-wise: result[i] = op(input[i]).
    fn unary_op(op: UnaryOp, input: &Self::Storage, layout: &Layout) -> Result<Self::Storage>;

    //  Reductions 

    /// Reduce along specific dimensions.
    /// If `dims` is empty, reduce over all elements.
    fn reduce_op(
        op: ReduceOp,
        input: &Self::Storage,
        layout: &Layout,
        dims: &[usize],
        keep_dim: bool,
    ) -> Result<Self::Storage>;

    //  Matrix multiplication 

    /// General matrix multiply: C = A @ B.
    /// Supports batched matmul for tensors with rank > 2.
    fn matmul(
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
    ) -> Result<Self::Storage>;

    //  Data movement 

    /// Make a contiguous copy of the storage following the given layout.
    /// If the layout is already contiguous, this may just clone the storage.
    fn to_contiguous(input: &Self::Storage, layout: &Layout) -> Result<Self::Storage>;

    /// Copy data from this storage to a Vec<f64> on the host (for inspection).
    fn to_f64_vec(input: &Self::Storage, layout: &Layout) -> Result<Vec<f64>>;

    //  Comparison ops 

    /// Element-wise comparison, returns a u8 storage (0 or 1).
    fn cmp_op(
        op: CmpOp,
        lhs: &Self::Storage,
        lhs_layout: &Layout,
        rhs: &Self::Storage,
        rhs_layout: &Layout,
    ) -> Result<Self::Storage>;

    //  Affine / fused ops (optional but useful) 

    /// Affine transform: result = input * mul + add.
    /// Used for normalization and other fused operations.
    fn affine(input: &Self::Storage, layout: &Layout, mul: f64, add: f64) -> Result<Self::Storage>;

    //  Indexing 

    /// Gather elements along a dimension using index tensor.
    fn index_select(
        input: &Self::Storage,
        input_layout: &Layout,
        indices: &Self::Storage,
        indices_layout: &Layout,
        dim: usize,
    ) -> Result<Self::Storage>;

    //  Powf 

    /// Element-wise power: result[i] = input[i] ^ exponent.
    fn powf(input: &Self::Storage, layout: &Layout, exponent: f64) -> Result<Self::Storage>;

    //  Clamp 

    /// Element-wise clamp: result[i] = clamp(input[i], min, max).
    fn clamp(input: &Self::Storage, layout: &Layout, min: f64, max: f64) -> Result<Self::Storage>;

    //  Where / conditional select 

    /// Element-wise conditional: result[i] = if mask[i] != 0 { on_true[i] } else { on_false[i] }.
    fn where_cond(
        mask: &Self::Storage,
        mask_layout: &Layout,
        on_true: &Self::Storage,
        on_true_layout: &Layout,
        on_false: &Self::Storage,
        on_false_layout: &Layout,
    ) -> Result<Self::Storage>;

    //  Gather 

    /// Gather elements along `dim` using `index` tensor.
    ///
    /// `output[i][j][k] = input[index[i][j][k]][j][k]`  (when dim=0)
    /// `output[i][j][k] = input[i][index[i][j][k]][k]`  (when dim=1)
    /// etc.
    ///
    /// `index` must have the same number of dimensions as `input`.
    fn gather(
        input: &Self::Storage,
        input_layout: &Layout,
        index: &Self::Storage,
        index_layout: &Layout,
        dim: usize,
    ) -> Result<Self::Storage>;

    //  Concatenation 

    /// Concatenate multiple storages along `dim` into a single contiguous storage.
    /// Each entry is (storage, layout) so non-contiguous inputs are handled correctly.
    /// `out_shape` is the pre-validated output shape.
    fn cat(
        inputs: &[(&Self::Storage, &Layout)],
        out_shape: &Shape,
        dim: usize,
    ) -> Result<Self::Storage>;

    //  Dtype conversion 

    /// Cast storage to a different dtype on-device (no host round-trip).
    ///
    /// The default implementation falls back to `to_f64_vec` + `from_f64_slice`,
    /// which involves a host round-trip. Backends should override this with
    /// a native on-device kernel when possible.
    fn cast(
        input: &Self::Storage,
        layout: &Layout,
        dtype: DType,
        device: &Self::Device,
    ) -> Result<Self::Storage> {
        let data = Self::to_f64_vec(input, layout)?;
        Self::from_f64_slice(&data, dtype, device)
    }
}
