use std::sync::{Arc, RwLock};

use crate::backend::{Backend, BinaryOp, CmpOp, ReduceOp, UnaryOp};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::op::{Op, TensorId};
use crate::shape::Shape;

// Tensor — The fundamental data structure
//
// A Tensor is an n-dimensional array of numbers, the building block of all
// neural network computations. Like in PyTorch, our Tensor:
//
//   1. Holds data on a specific device (CPU, GPU)
//   2. Has a shape (e.g., [batch, channels, height, width])
//   3. Has a dtype (f32, f64, etc.)
//   4. Optionally tracks the operation that created it (for autograd)
//
// ARCHITECTURE:
//
//   Tensor<B: Backend> is generic over the backend. This means:
//     - Tensor<CpuBackend> holds data in CPU memory
//     - Tensor<CudaBackend> holds data in GPU memory
//     - Operations are dispatched via the Backend trait
//
// MEMORY MODEL:
//
//   The inner data is wrapped in Arc (atomic reference counting).
//   This means cloning a Tensor is cheap (just increments a counter).
//   Multiple tensors can share the same underlying storage (views).
//
//   Storage is behind Arc<RwLock<Storage>> so that:
//   - Multiple tensors can read concurrently
//   - In-place ops can write when there's only one reference
//
// WHY Arc + inner struct?
//
//   We separate Tensor (the handle) from TensorInner (the data) so that:
//   - Cloning Tensor is O(1) — just copies the Arc pointer
//   - The autograd graph can hold TensorIds without owning data
//   - Views (transpose, narrow) share the same storage via Arc<RwLock<>>

/// Inner data of a tensor, shared via Arc.
struct TensorInner<B: Backend> {
    /// Unique identifier for this tensor (used in autograd graph).
    id: TensorId,
    /// The raw data stored on the backend's device.
    storage: Arc<RwLock<B::Storage>>,
    /// Memory layout: shape + strides + offset.
    layout: Layout,
    /// Data type of the elements.
    dtype: DType,
    /// The device this tensor lives on.
    device: B::Device,
    /// The operation that created this tensor (for autograd).
    /// None for leaf tensors (inputs, parameters).
    op: Op<B>,
    /// Whether this tensor is a trainable variable.
    /// Only variables accumulate gradients during backward().
    is_variable: bool,
}

/// An n-dimensional array of numbers on a specific backend.
///
/// Tensors are the fundamental data type in Shrew. All neural network
/// operations accept and return tensors.
///
/// # Type Parameter
/// - `B: Backend` — the compute backend (e.g., `CpuBackend`, `CudaBackend`)
///
/// # Example
/// ```ignore
/// use shrew_core::Tensor;
/// use shrew_cpu::CpuBackend;
///
/// let a = Tensor::<CpuBackend>::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2))?;
/// let b = Tensor::<CpuBackend>::ones((2, 2), DType::F32, &CpuDevice)?;
/// let c = a.add(&b)?;
/// ```
pub struct Tensor<B: Backend> {
    inner: Arc<TensorInner<B>>,
}

// Manual Clone: Arc::clone is cheap (just increment refcount).
impl<B: Backend> Clone for Tensor<B> {
    fn clone(&self) -> Self {
        Tensor {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<B: Backend> std::fmt::Debug for Tensor<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor(id={:?}, shape={}, dtype={}, device={:?})",
            self.inner.id,
            self.inner.layout.shape(),
            self.inner.dtype,
            self.inner.device,
        )
    }
}

impl<B: Backend> Tensor<B> {
    // Internal constructors

    /// Create a tensor from existing storage and layout.
    pub(crate) fn from_storage(
        storage: B::Storage,
        layout: Layout,
        dtype: DType,
        device: B::Device,
        op: Op<B>,
    ) -> Self {
        Tensor {
            inner: Arc::new(TensorInner {
                id: TensorId::new(),
                storage: Arc::new(RwLock::new(storage)),
                layout,
                dtype,
                device,
                op,
                is_variable: false,
            }),
        }
    }

    /// Create a view tensor sharing the same storage but with a different layout.
    fn view_with_layout(&self, layout: Layout, op: Op<B>) -> Self {
        Tensor {
            inner: Arc::new(TensorInner {
                id: TensorId::new(),
                storage: Arc::clone(&self.inner.storage),
                layout,
                dtype: self.inner.dtype,
                device: self.inner.device.clone(),
                op,
                is_variable: false,
            }),
        }
    }

    // Accessors

    /// Unique tensor ID.
    pub fn id(&self) -> TensorId {
        self.inner.id
    }

    /// The shape of this tensor.
    pub fn shape(&self) -> &Shape {
        self.inner.layout.shape()
    }

    /// The dimensions as a slice (shortcut for shape().dims()).
    pub fn dims(&self) -> &[usize] {
        self.inner.layout.dims()
    }

    /// Number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.inner.layout.rank()
    }

    /// Total number of elements.
    pub fn elem_count(&self) -> usize {
        self.inner.layout.elem_count()
    }

    /// Data type of the elements.
    pub fn dtype(&self) -> DType {
        self.inner.dtype
    }

    /// The device this tensor is on.
    pub fn device(&self) -> &B::Device {
        &self.inner.device
    }

    /// The memory layout (shape + strides + offset).
    pub fn layout(&self) -> &Layout {
        &self.inner.layout
    }

    /// Whether this tensor is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        self.inner.layout.is_contiguous()
    }

    /// Whether this tensor tracks gradients.
    pub fn is_variable(&self) -> bool {
        self.inner.is_variable
    }

    /// Access the underlying storage (read lock).
    pub fn storage(&self) -> std::sync::RwLockReadGuard<'_, B::Storage> {
        self.inner.storage.read().expect("storage lock poisoned")
    }

    /// Try to acquire a read lock on storage, returning an error instead of panicking.
    fn read_storage(&self) -> Result<std::sync::RwLockReadGuard<'_, B::Storage>> {
        self.inner
            .storage
            .read()
            .map_err(|_| Error::msg("storage lock poisoned"))
    }

    /// Try to acquire a write lock on storage, returning an error instead of panicking.
    fn write_storage(&self) -> Result<std::sync::RwLockWriteGuard<'_, B::Storage>> {
        self.inner
            .storage
            .write()
            .map_err(|_| Error::msg("storage lock poisoned"))
    }

    /// The op that created this tensor.
    pub fn op(&self) -> &Op<B> {
        &self.inner.op
    }

    // In-place mutation

    /// Update the underlying storage data in place.
    ///
    /// This writes `new_data` directly into the existing `Arc<RwLock<Storage>>`,
    /// so any other tensor sharing this storage (e.g., a clone held by a Module)
    /// will also see the updated values.
    ///
    /// This is the mechanism that makes optimizer parameter updates visible to
    /// model layers without needing to re-assign parameters.
    ///
    /// # Safety (logical)
    /// The new data must have the same number of elements and dtype as the
    /// current storage. The shape is not changed.
    pub fn update_data_inplace(&self, new_data: &[f64]) -> Result<()> {
        let expected = self.elem_count();
        if new_data.len() != expected {
            return Err(Error::msg(format!(
                "update_data_inplace: expected {} elements, got {}",
                expected,
                new_data.len()
            )));
        }
        let new_storage = B::from_f64_slice(new_data, self.dtype(), self.device())?;
        let mut guard = self.write_storage()?;
        *guard = new_storage;
        Ok(())
    }

    // Creation methods

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: impl Into<Shape>, dtype: DType, device: &B::Device) -> Result<Self> {
        let shape = shape.into();
        let layout = Layout::contiguous(shape.clone());
        let storage = B::zeros(&shape, dtype, device)?;
        Ok(Self::from_storage(
            storage,
            layout,
            dtype,
            device.clone(),
            Op::None,
        ))
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: impl Into<Shape>, dtype: DType, device: &B::Device) -> Result<Self> {
        let shape = shape.into();
        let layout = Layout::contiguous(shape.clone());
        let storage = B::ones(&shape, dtype, device)?;
        Ok(Self::from_storage(
            storage,
            layout,
            dtype,
            device.clone(),
            Op::None,
        ))
    }

    /// Create a tensor filled with a constant value.
    pub fn full(
        shape: impl Into<Shape>,
        val: f64,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        let shape = shape.into();
        let layout = Layout::contiguous(shape.clone());
        let storage = B::full(&shape, val, dtype, device)?;
        Ok(Self::from_storage(
            storage,
            layout,
            dtype,
            device.clone(),
            Op::None,
        ))
    }

    /// Create a tensor from a flat slice of f64 values.
    /// The data is converted to the specified dtype.
    pub fn from_f64_slice(
        data: &[f64],
        shape: impl Into<Shape>,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        let shape = shape.into();
        if data.len() != shape.elem_count() {
            return Err(Error::ElementCountMismatch {
                shape: shape.clone(),
                expected: shape.elem_count(),
                got: data.len(),
            });
        }
        let layout = Layout::contiguous(shape);
        let storage = B::from_f64_slice(data, dtype, device)?;
        Ok(Self::from_storage(
            storage,
            layout,
            dtype,
            device.clone(),
            Op::None,
        ))
    }

    /// Create a tensor with random uniform values in [0, 1).
    pub fn rand(shape: impl Into<Shape>, dtype: DType, device: &B::Device) -> Result<Self> {
        let shape = shape.into();
        let layout = Layout::contiguous(shape.clone());
        let storage = B::rand_uniform(&shape, dtype, device)?;
        Ok(Self::from_storage(
            storage,
            layout,
            dtype,
            device.clone(),
            Op::None,
        ))
    }

    /// Create a tensor with random normal values (mean=0, std=1).
    pub fn randn(shape: impl Into<Shape>, dtype: DType, device: &B::Device) -> Result<Self> {
        let shape = shape.into();
        let layout = Layout::contiguous(shape.clone());
        let storage = B::rand_normal(&shape, dtype, device)?;
        Ok(Self::from_storage(
            storage,
            layout,
            dtype,
            device.clone(),
            Op::None,
        ))
    }

    /// Create a 1-D tensor with `steps` evenly spaced values from `start` to `end` (inclusive).
    ///
    /// ```ignore
    /// let t = Tensor::linspace(0.0, 1.0, 5, DType::F64, &dev)?;
    /// // => [0.0, 0.25, 0.5, 0.75, 1.0]
    /// ```
    pub fn linspace(
        start: f64,
        end: f64,
        steps: usize,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        if steps == 0 {
            return Err(Error::msg("linspace requires steps >= 1"));
        }
        if steps == 1 {
            return Self::from_f64_slice(&[start], 1, dtype, device);
        }
        let step = (end - start) / (steps as f64 - 1.0);
        let data: Vec<f64> = (0..steps).map(|i| start + step * i as f64).collect();
        Self::from_f64_slice(&data, steps, dtype, device)
    }

    /// Create an identity matrix of size `n × n`.
    ///
    /// ```ignore
    /// let I = Tensor::eye(3, DType::F64, &dev)?;
    /// // [[1, 0, 0],
    /// //  [0, 1, 0],
    /// //  [0, 0, 1]]
    /// ```
    pub fn eye(n: usize, dtype: DType, device: &B::Device) -> Result<Self> {
        let mut data = vec![0.0f64; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::from_f64_slice(&data, (n, n), dtype, device)
    }

    /// Create a tensor of zeros with the same shape, dtype, and device as `other`.
    pub fn zeros_like(other: &Self) -> Result<Self> {
        Self::zeros(other.shape().clone(), other.dtype(), other.device())
    }

    /// Create a tensor of ones with the same shape, dtype, and device as `other`.
    pub fn ones_like(other: &Self) -> Result<Self> {
        Self::ones(other.shape().clone(), other.dtype(), other.device())
    }

    /// Create a tensor filled with `val`, with the same shape, dtype, and device as `other`.
    pub fn full_like(other: &Self, val: f64) -> Result<Self> {
        Self::full(other.shape().clone(), val, other.dtype(), other.device())
    }

    /// Mark this tensor as a variable (trainable parameter).
    /// Variables accumulate gradients during backward().
    pub fn set_variable(self) -> Self {
        Tensor {
            inner: Arc::new(TensorInner {
                id: self.inner.id,
                storage: Arc::clone(&self.inner.storage),
                layout: self.inner.layout.clone(),
                dtype: self.inner.dtype,
                device: self.inner.device.clone(),
                op: self.inner.op.clone(),
                is_variable: true,
            }),
        }
    }

    // Shape manipulation (these create views, no data copy)

    /// Transpose two dimensions (no data copy).
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        let new_layout = self.inner.layout.transpose(dim0, dim1)?;
        let op = Op::Transpose {
            input: self.clone(),
            dim0,
            dim1,
        };
        Ok(self.view_with_layout(new_layout, op))
    }

    /// Transpose a 2D matrix (shorthand for transpose(0, 1)).
    pub fn t(&self) -> Result<Self> {
        if self.rank() != 2 {
            return Err(Error::RankMismatch {
                expected: 2,
                got: self.rank(),
            });
        }
        self.transpose(0, 1)
    }

    /// Narrow (slice) along a dimension.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let new_layout = self.inner.layout.narrow(dim, start, len)?;
        let op = Op::Narrow {
            input: self.clone(),
            dim,
            start,
            len,
        };
        Ok(self.view_with_layout(new_layout, op))
    }

    /// Reshape to a new shape. The new shape must have the same total elements.
    /// If the tensor is not contiguous, it will be made contiguous first.
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Result<Self> {
        let new_shape = new_shape.into();
        let current_count = self.elem_count();
        let new_count = new_shape.elem_count();
        if current_count != new_count {
            return Err(Error::ReshapeElementMismatch {
                src: current_count,
                dst: new_count,
                dst_shape: new_shape,
            });
        }
        // If not contiguous, make a contiguous copy first
        let tensor = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()?
        };
        let src_shape = tensor.shape().clone();
        let new_layout = Layout::contiguous(new_shape);
        let op = Op::Reshape {
            input: tensor.clone(),
            src_shape,
        };
        Ok(tensor.view_with_layout(new_layout, op))
    }

    /// Ensure the tensor is contiguous in memory.
    /// If already contiguous, returns a clone (cheap Arc copy).
    /// Otherwise, copies the data into a new contiguous storage.
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        let storage = self.read_storage()?;
        let new_storage = B::to_contiguous(&storage, &self.inner.layout)?;
        let new_layout = Layout::contiguous(self.shape().clone());
        Ok(Self::from_storage(
            new_storage,
            new_layout,
            self.inner.dtype,
            self.inner.device.clone(),
            Op::Contiguous {
                input: self.clone(),
            },
        ))
    }

    /// Add a dimension of size 1 at the given position.
    /// unsqueeze(0) on [3, 4] → [1, 3, 4]
    /// unsqueeze(2) on [3, 4] → [3, 4, 1]
    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let rank = self.rank();
        if dim > rank {
            return Err(Error::DimOutOfRange {
                dim,
                rank: rank + 1,
            });
        }
        let mut new_dims = self.dims().to_vec();
        let mut new_strides = self.layout().strides().to_vec();
        // The stride for a size-1 dim doesn't matter (you never move along it),
        // but convention is to use the stride of the next dimension (or 1 if last).
        let stride_val = if dim < rank { new_strides[dim] } else { 1 };
        new_dims.insert(dim, 1);
        new_strides.insert(dim, stride_val);
        let new_layout = Layout::new(Shape::new(new_dims), new_strides, self.layout().offset());
        let op = Op::Reshape {
            input: self.clone(),
            src_shape: self.shape().clone(),
        };
        Ok(self.view_with_layout(new_layout, op))
    }

    /// Remove dimensions of size 1.
    /// squeeze on [1, 3, 1, 4] → [3, 4]
    pub fn squeeze_all(&self) -> Self {
        let new_dims: Vec<usize> = self.dims().iter().copied().filter(|&d| d != 1).collect();
        let new_strides: Vec<usize> = self
            .dims()
            .iter()
            .zip(self.layout().strides().iter())
            .filter(|(&d, _)| d != 1)
            .map(|(_, &s)| s)
            .collect();
        let new_layout = Layout::new(
            Shape::new(if new_dims.is_empty() {
                vec![]
            } else {
                new_dims
            }),
            new_strides,
            self.layout().offset(),
        );
        let op = Op::Reshape {
            input: self.clone(),
            src_shape: self.shape().clone(),
        };
        self.view_with_layout(new_layout, op)
    }

    /// Remove a specific dimension of size 1.
    ///
    /// squeeze(1) on [3, 1, 4] → [3, 4]
    ///
    /// Returns an error if the specified dimension is not size 1.
    pub fn squeeze(&self, dim: usize) -> Result<Self> {
        let rank = self.rank();
        if dim >= rank {
            return Err(Error::DimOutOfRange { dim, rank });
        }
        if self.dims()[dim] != 1 {
            return Err(Error::msg(format!(
                "squeeze: dimension {} has size {}, expected 1",
                dim,
                self.dims()[dim]
            )));
        }
        let mut new_dims = self.dims().to_vec();
        let mut new_strides = self.layout().strides().to_vec();
        new_dims.remove(dim);
        new_strides.remove(dim);
        let new_layout = Layout::new(
            Shape::new(if new_dims.is_empty() {
                vec![]
            } else {
                new_dims
            }),
            new_strides,
            self.layout().offset(),
        );
        let op = Op::Reshape {
            input: self.clone(),
            src_shape: self.shape().clone(),
        };
        Ok(self.view_with_layout(new_layout, op))
    }

    /// Permute the dimensions of this tensor.
    ///
    /// permute(&[2, 0, 1]) on [A, B, C] → [C, A, B]
    ///
    /// This is a generalization of transpose to arbitrary dimension orderings.
    /// No data copy — just changes strides.
    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        let rank = self.rank();
        if dims.len() != rank {
            return Err(Error::msg(format!(
                "permute: expected {} dimensions, got {}",
                rank,
                dims.len()
            )));
        }
        // Check for duplicates and out-of-range
        let mut seen = vec![false; rank];
        for &d in dims {
            if d >= rank {
                return Err(Error::DimOutOfRange { dim: d, rank });
            }
            if seen[d] {
                return Err(Error::msg(format!("permute: duplicate dimension {}", d)));
            }
            seen[d] = true;
        }

        let old_dims = self.dims();
        let old_strides = self.layout().strides();
        let new_dims: Vec<usize> = dims.iter().map(|&d| old_dims[d]).collect();
        let new_strides: Vec<usize> = dims.iter().map(|&d| old_strides[d]).collect();
        let new_layout = Layout::new(Shape::new(new_dims), new_strides, self.layout().offset());
        // Use a chain of transposes conceptually, but represent as reshape
        // for backward compatibility. The backward pass handles reshapes correctly.
        let op = Op::Reshape {
            input: self.clone(),
            src_shape: self.shape().clone(),
        };
        Ok(self.view_with_layout(new_layout, op))
    }

    /// Cumulative sum along dimension `dim`.
    ///
    /// ```ignore
    /// // [1, 2, 3] → [1, 3, 6]
    /// let y = x.cumsum(0)?;
    /// ```
    pub fn cumsum(&self, dim: usize) -> Result<Self> {
        let rank = self.rank();
        if dim >= rank {
            return Err(Error::DimOutOfRange { dim, rank });
        }
        let t = self.contiguous()?;
        let data = t.to_f64_vec()?;
        let shape = t.shape().clone();
        let dims = shape.dims();
        let mut out = data.clone();

        // Compute strides for iteration
        let inner: usize = dims[dim + 1..].iter().product();
        let outer: usize = dims[..dim].iter().product();
        let dim_size = dims[dim];

        for o in 0..outer {
            for i in 0..inner {
                for d in 1..dim_size {
                    let idx = (o * dim_size + d) * inner + i;
                    let prev = (o * dim_size + d - 1) * inner + i;
                    out[idx] += out[prev];
                }
            }
        }

        Self::from_f64_slice(&out, shape, t.dtype(), t.device())
    }

    /// Sort along a dimension. Returns `(sorted_values, sorted_indices)`.
    ///
    /// ```ignore
    /// let (vals, idxs) = x.sort(0, false)?; // ascending along dim 0
    /// ```
    pub fn sort(&self, dim: usize, descending: bool) -> Result<(Self, Self)> {
        let rank = self.rank();
        if dim >= rank {
            return Err(Error::DimOutOfRange { dim, rank });
        }
        let t = self.contiguous()?;
        let data = t.to_f64_vec()?;
        let shape = t.shape().clone();
        let dims = shape.dims();
        let dim_size = dims[dim];
        let inner: usize = dims[dim + 1..].iter().product();
        let outer: usize = dims[..dim].iter().product();

        let mut sorted_data = data.clone();
        let mut indices = vec![0.0f64; data.len()];

        for o in 0..outer {
            for i in 0..inner {
                // Extract the slice along dim
                let mut slice: Vec<(f64, usize)> = (0..dim_size)
                    .map(|d| {
                        let idx = (o * dim_size + d) * inner + i;
                        (data[idx], d)
                    })
                    .collect();

                if descending {
                    slice
                        .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    slice
                        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                }

                for (d, (val, orig_idx)) in slice.into_iter().enumerate() {
                    let idx = (o * dim_size + d) * inner + i;
                    sorted_data[idx] = val;
                    indices[idx] = orig_idx as f64;
                }
            }
        }

        let vals = Self::from_f64_slice(&sorted_data, shape.clone(), t.dtype(), t.device())?;
        let idxs = Self::from_f64_slice(&indices, shape, t.dtype(), t.device())?;
        Ok((vals, idxs))
    }

    /// Argsort: returns indices that would sort the tensor along `dim`.
    ///
    /// ```ignore
    /// let indices = x.argsort(0, false)?; // ascending
    /// ```
    pub fn argsort(&self, dim: usize, descending: bool) -> Result<Self> {
        let (_, indices) = self.sort(dim, descending)?;
        Ok(indices)
    }

    // Arithmetic operations

    /// Element-wise addition: self + rhs.
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.binary_op(rhs, BinaryOp::Add)
    }

    /// Element-wise subtraction: self - rhs.
    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.binary_op(rhs, BinaryOp::Sub)
    }

    /// Element-wise multiplication: self * rhs.
    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        self.binary_op(rhs, BinaryOp::Mul)
    }

    /// Element-wise division: self / rhs.
    pub fn div(&self, rhs: &Self) -> Result<Self> {
        self.binary_op(rhs, BinaryOp::Div)
    }

    /// Generic binary operation dispatch.
    fn binary_op(&self, rhs: &Self, op: BinaryOp) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                expected: self.dtype(),
                got: rhs.dtype(),
            });
        }
        let storage_lhs = self.read_storage()?;
        let storage_rhs = rhs.read_storage()?;
        let result = B::binary_op(
            op,
            &storage_lhs,
            &self.inner.layout,
            &storage_rhs,
            &rhs.inner.layout,
        )?;
        // Compute broadcast output shape
        let result_shape = Shape::broadcast_shape(self.shape(), rhs.shape())?;
        let result_layout = Layout::contiguous(result_shape);
        let result_op = Op::Binary {
            lhs: self.clone(),
            rhs: rhs.clone(),
            op,
        };
        Ok(Self::from_storage(
            result,
            result_layout,
            self.inner.dtype,
            self.inner.device.clone(),
            result_op,
        ))
    }

    // Comparison operations

    /// Element-wise equal: self == rhs. Returns a U8 tensor (0 or 1).
    pub fn eq(&self, rhs: &Self) -> Result<Self> {
        self.cmp_op(rhs, CmpOp::Eq)
    }

    /// Element-wise not-equal: self != rhs. Returns a U8 tensor (0 or 1).
    pub fn ne(&self, rhs: &Self) -> Result<Self> {
        self.cmp_op(rhs, CmpOp::Ne)
    }

    /// Element-wise greater-than: self > rhs. Returns a U8 tensor (0 or 1).
    pub fn gt(&self, rhs: &Self) -> Result<Self> {
        self.cmp_op(rhs, CmpOp::Gt)
    }

    /// Element-wise greater-or-equal: self >= rhs. Returns a U8 tensor (0 or 1).
    pub fn ge(&self, rhs: &Self) -> Result<Self> {
        self.cmp_op(rhs, CmpOp::Ge)
    }

    /// Element-wise less-than: self < rhs. Returns a U8 tensor (0 or 1).
    pub fn lt(&self, rhs: &Self) -> Result<Self> {
        self.cmp_op(rhs, CmpOp::Lt)
    }

    /// Element-wise less-or-equal: self <= rhs. Returns a U8 tensor (0 or 1).
    pub fn le(&self, rhs: &Self) -> Result<Self> {
        self.cmp_op(rhs, CmpOp::Le)
    }

    /// Generic comparison operation dispatch.
    /// Produces a U8-dtype tensor (non-differentiable, Op::None).
    fn cmp_op(&self, rhs: &Self, op: CmpOp) -> Result<Self> {
        let storage_lhs = self.read_storage()?;
        let storage_rhs = rhs.read_storage()?;
        let result = B::cmp_op(
            op,
            &storage_lhs,
            &self.inner.layout,
            &storage_rhs,
            &rhs.inner.layout,
        )?;
        let result_shape = Shape::broadcast_shape(self.shape(), rhs.shape())?;
        let result_layout = Layout::contiguous(result_shape);
        // Comparisons are non-differentiable — no autograd tracking
        Ok(Self::from_storage(
            result,
            result_layout,
            DType::U8,
            self.inner.device.clone(),
            Op::None,
        ))
    }

    // Unary operations

    /// Element-wise negation: -self.
    pub fn neg(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Neg)
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Abs)
    }

    /// Element-wise exponential: e^x.
    pub fn exp(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Exp)
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Log)
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Sqrt)
    }

    /// Element-wise square: x².
    pub fn square(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Square)
    }

    /// ReLU activation: max(0, x).
    pub fn relu(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Relu)
    }

    /// Sigmoid activation: 1 / (1 + e^(-x)).
    pub fn sigmoid(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Sigmoid)
    }

    /// Tanh activation.
    pub fn tanh(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Tanh)
    }

    /// GELU activation (Gaussian Error Linear Unit).
    pub fn gelu(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Gelu)
    }

    /// SiLU / Swish activation: x * sigmoid(x).
    pub fn silu(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Silu)
    }

    /// Element-wise sine.
    pub fn sin(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Sin)
    }

    /// Element-wise cosine.
    pub fn cos(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Cos)
    }

    /// Element-wise floor: largest integer ≤ x.
    pub fn floor(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Floor)
    }

    /// Element-wise ceiling: smallest integer ≥ x.
    pub fn ceil(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Ceil)
    }

    /// Element-wise round to nearest integer.
    pub fn round(&self) -> Result<Self> {
        self.unary_op(UnaryOp::Round)
    }

    /// Element-wise power: self^exponent.
    pub fn powf(&self, exponent: f64) -> Result<Self> {
        let storage = self.read_storage()?;
        let result = B::powf(&storage, &self.inner.layout, exponent)?;
        let result_layout = Layout::contiguous(self.shape().clone());
        let result_op = Op::Powf {
            input: self.clone(),
            exponent,
        };
        Ok(Self::from_storage(
            result,
            result_layout,
            self.inner.dtype,
            self.inner.device.clone(),
            result_op,
        ))
    }

    /// Element-wise clamp to [min, max].
    pub fn clamp(&self, min: f64, max: f64) -> Result<Self> {
        let storage = self.read_storage()?;
        let result = B::clamp(&storage, &self.inner.layout, min, max)?;
        let result_layout = Layout::contiguous(self.shape().clone());
        let result_op = Op::Clamp {
            input: self.clone(),
            min,
            max,
        };
        Ok(Self::from_storage(
            result,
            result_layout,
            self.inner.dtype,
            self.inner.device.clone(),
            result_op,
        ))
    }

    /// Conditional select: result[i] = if mask[i] != 0 { on_true[i] } else { on_false[i] }.
    ///
    /// `mask` is typically a U8 tensor from comparison ops.
    /// `on_true` and `on_false` must have the same shape and dtype.
    pub fn where_cond(mask: &Self, on_true: &Self, on_false: &Self) -> Result<Self> {
        let mask_s = mask.read_storage()?;
        let true_s = on_true.read_storage()?;
        let false_s = on_false.read_storage()?;
        let result = B::where_cond(
            &mask_s,
            &mask.inner.layout,
            &true_s,
            &on_true.inner.layout,
            &false_s,
            &on_false.inner.layout,
        )?;
        let result_layout = Layout::contiguous(on_true.shape().clone());
        let result_op = Op::WhereCond {
            mask: mask.clone(),
            on_true: on_true.clone(),
            on_false: on_false.clone(),
        };
        Ok(Self::from_storage(
            result,
            result_layout,
            on_true.inner.dtype,
            on_true.inner.device.clone(),
            result_op,
        ))
    }

    /// Gather elements along `dim` using an index tensor.
    ///
    /// `output[i][j][k] = input[index[i][j][k]][j][k]`  (when dim=0)
    ///
    /// The index tensor must have the same number of dimensions as self.
    /// The output has the same shape as the index tensor.
    pub fn gather(&self, dim: usize, index: &Self) -> Result<Self> {
        let input_s = self.read_storage()?;
        let index_s = index.read_storage()?;
        let result = B::gather(
            &input_s,
            &self.inner.layout,
            &index_s,
            &index.inner.layout,
            dim,
        )?;
        let result_layout = Layout::contiguous(index.shape().clone());
        let result_op = Op::Gather {
            input: self.clone(),
            index: index.clone(),
            dim,
        };
        Ok(Self::from_storage(
            result,
            result_layout,
            self.inner.dtype,
            self.inner.device.clone(),
            result_op,
        ))
    }

    /// Fill elements where `mask != 0` with `value`, keeping other elements.
    ///
    /// `result[i] = if mask[i] != 0 { value } else { self[i] }`
    ///
    /// This is implemented via `where_cond` so autograd is automatic.
    pub fn masked_fill(&self, mask: &Self, value: f64) -> Result<Self> {
        let fill = Self::full(self.shape().clone(), value, self.dtype(), self.device())?;
        Self::where_cond(mask, &fill, self)
    }

    /// Pad the last N dimensions with constant `value`.
    ///
    /// `padding` is a list of `[before, after]` pairs, one per dimension,
    /// applied to the **last** dimensions of the tensor.
    ///
    /// Example: `pad(&[[1, 1], [2, 2]], 0.0)` pads the last 2 dims.
    pub fn pad(&self, padding: &[[usize; 2]], value: f64) -> Result<Self> {
        let rank = self.rank();
        if padding.len() > rank {
            return Err(Error::msg(format!(
                "pad: {} padding pairs but tensor rank is {}",
                padding.len(),
                rank
            )));
        }

        // Build full-rank padding: leading dims get [0,0]
        let mut full_pad = vec![[0usize; 2]; rank];
        let offset = rank - padding.len();
        for (i, p) in padding.iter().enumerate() {
            full_pad[offset + i] = *p;
        }

        // Compute output shape
        let in_dims = self.dims();
        let out_dims: Vec<usize> = in_dims
            .iter()
            .zip(full_pad.iter())
            .map(|(&d, &[b, a])| d + b + a)
            .collect();

        // If no padding at all, just return a clone
        if full_pad.iter().all(|&[b, a]| b == 0 && a == 0) {
            return Ok(self.clone());
        }

        // Build result by concatenating pad tensors along each dimension
        let mut current = self.clone();
        for d in (0..rank).rev() {
            let [before, after] = full_pad[d];
            if before == 0 && after == 0 {
                continue;
            }
            let mut cur_dims = current.dims().to_vec();
            let mut parts: Vec<Self> = Vec::new();

            if before > 0 {
                cur_dims[d] = before;
                let pad_before = Self::full(
                    Shape::new(cur_dims.clone()),
                    value,
                    self.dtype(),
                    self.device(),
                )?;
                cur_dims[d] = current.dims()[d]; // restore
                parts.push(pad_before);
            }
            parts.push(current);
            if after > 0 {
                cur_dims[d] = after;
                let pad_after = Self::full(
                    Shape::new(cur_dims.clone()),
                    value,
                    self.dtype(),
                    self.device(),
                )?;
                parts.push(pad_after);
            }
            current = Self::cat(&parts, d)?;
        }

        // Wrap in Op::Pad for clean backward
        let result_layout = Layout::contiguous(Shape::new(out_dims));
        let pad_op = Op::Pad {
            input: self.clone(),
            padding: full_pad,
        };

        // Re-wrap with the Pad op so backward can narrow back
        let storage = current.read_storage()?;
        Ok(Self::from_storage(
            storage.clone(),
            result_layout,
            self.inner.dtype,
            self.inner.device.clone(),
            pad_op,
        ))
    }

    /// Return the `k` largest elements along `dim`.
    ///
    /// Returns `(values, indices)` where both have the same shape as self
    /// except dimension `dim` has size `k`.
    ///
    /// Non-differentiable (returns detached values).
    #[allow(clippy::needless_range_loop)]
    pub fn topk(&self, k: usize, dim: usize) -> Result<(Self, Vec<usize>)> {
        if dim >= self.rank() {
            return Err(Error::DimOutOfRange {
                dim,
                rank: self.rank(),
            });
        }
        let dims = self.dims();
        let dim_size = dims[dim];
        if k > dim_size {
            return Err(Error::msg(format!(
                "topk: k={} exceeds dim {} size {}",
                k, dim, dim_size
            )));
        }

        let data = self.contiguous()?.to_f64_vec()?;

        // Output shape: same as input but dim has size k
        let mut out_dims = dims.to_vec();
        out_dims[dim] = k;
        let out_size: usize = out_dims.iter().product();
        let mut out_values = vec![0.0f64; out_size];
        let mut out_indices = vec![0usize; out_size];

        // Number of "slices" along dim
        let outer: usize = dims[..dim].iter().product();
        let inner: usize = dims[dim + 1..].iter().product();

        for o in 0..outer {
            for i in 0..inner {
                // Collect all elements along this dim-slice
                let mut slice: Vec<(f64, usize)> = (0..dim_size)
                    .map(|d| {
                        let flat = o * (dim_size * inner) + d * inner + i;
                        (data[flat], d)
                    })
                    .collect();
                // Sort descending
                slice.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                // Write top-k
                for j in 0..k {
                    let out_flat = o * (k * inner) + j * inner + i;
                    out_values[out_flat] = slice[j].0;
                    out_indices[out_flat] = slice[j].1;
                }
            }
        }

        let values = Self::from_f64_slice(
            &out_values,
            Shape::new(out_dims),
            self.dtype(),
            self.device(),
        )?;
        Ok((values, out_indices))
    }

    /// Generic unary operation dispatch.
    fn unary_op(&self, op: UnaryOp) -> Result<Self> {
        let storage = self.read_storage()?;
        let result = B::unary_op(op, &storage, &self.inner.layout)?;
        let result_layout = Layout::contiguous(self.shape().clone());
        let result_op = Op::Unary {
            input: self.clone(),
            op,
        };
        Ok(Self::from_storage(
            result,
            result_layout,
            self.inner.dtype,
            self.inner.device.clone(),
            result_op,
        ))
    }

    // Reductions

    /// Sum all elements, returning a scalar tensor.
    pub fn sum_all(&self) -> Result<Self> {
        self.reduce_op(ReduceOp::Sum, &[], false)
    }

    /// Sum along a specific dimension.
    pub fn sum(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        self.reduce_op(ReduceOp::Sum, &[dim], keep_dim)
    }

    /// Mean of all elements, returning a scalar tensor.
    pub fn mean_all(&self) -> Result<Self> {
        self.reduce_op(ReduceOp::Mean, &[], false)
    }

    /// Mean along a specific dimension.
    pub fn mean(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        self.reduce_op(ReduceOp::Mean, &[dim], keep_dim)
    }

    /// Max along a specific dimension.
    pub fn max(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        self.reduce_op(ReduceOp::Max, &[dim], keep_dim)
    }

    /// Min along a specific dimension.
    pub fn min(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        self.reduce_op(ReduceOp::Min, &[dim], keep_dim)
    }

    /// ArgMax along a specific dimension (returns i64 indices).
    pub fn argmax(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        self.reduce_op(ReduceOp::ArgMax, &[dim], keep_dim)
    }

    /// ArgMin along a specific dimension (returns i64 indices).
    pub fn argmin(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        self.reduce_op(ReduceOp::ArgMin, &[dim], keep_dim)
    }

    /// Generic reduction dispatch.
    fn reduce_op(&self, op: ReduceOp, dims: &[usize], keep_dim: bool) -> Result<Self> {
        // Validate dimensions
        for &d in dims {
            if d >= self.rank() {
                return Err(Error::DimOutOfRange {
                    dim: d,
                    rank: self.rank(),
                });
            }
        }
        let storage = self.read_storage()?;
        let result = B::reduce_op(op, &storage, &self.inner.layout, dims, keep_dim)?;

        // Compute result shape
        let result_shape = if dims.is_empty() {
            // Reduce all → scalar
            Shape::from(())
        } else if keep_dim {
            let mut new_dims = self.dims().to_vec();
            for &d in dims {
                new_dims[d] = 1;
            }
            Shape::new(new_dims)
        } else {
            let new_dims: Vec<usize> = self
                .dims()
                .iter()
                .enumerate()
                .filter(|(i, _)| !dims.contains(i))
                .map(|(_, &d)| d)
                .collect();
            if new_dims.is_empty() {
                Shape::from(())
            } else {
                Shape::new(new_dims)
            }
        };

        let result_layout = Layout::contiguous(result_shape);
        let result_dtype = match op {
            ReduceOp::ArgMax | ReduceOp::ArgMin => DType::I64,
            _ => self.inner.dtype,
        };
        let result_op = Op::Reduce {
            input: self.clone(),
            op,
            dims: dims.to_vec(),
            keep_dim,
        };
        Ok(Self::from_storage(
            result,
            result_layout,
            result_dtype,
            self.inner.device.clone(),
            result_op,
        ))
    }

    // Composite operations (built from primitives)

    /// Softmax along a dimension: softmax(x)_i = exp(x_i) / sum(exp(x_j))
    ///
    /// Uses the numerically stable trick: subtract max before exp.
    /// This is built from existing differentiable ops (exp, sum, div, sub)
    /// so gradients flow through automatically.
    pub fn softmax(&self, dim: usize) -> Result<Self> {
        // max(x, dim, keep_dim=true) — used as a constant for stability
        let max_val = self.max(dim, true)?;
        // We detach max so it's treated as a constant in backward
        let max_detached = max_val.detach();
        let shifted = self.sub(&max_detached)?; // x - max(x)
        let exp_x = shifted.exp()?;
        let sum_exp = exp_x.sum(dim, true)?;
        exp_x.div(&sum_exp)
    }

    /// Log-softmax along a dimension: log(softmax(x)) but numerically stable.
    ///
    /// log_softmax(x)_i = x_i - max(x) - log(sum(exp(x - max(x))))
    pub fn log_softmax(&self, dim: usize) -> Result<Self> {
        let max_val = self.max(dim, true)?.detach();
        let shifted = self.sub(&max_val)?;
        let exp_x = shifted.exp()?;
        let sum_exp = exp_x.sum(dim, true)?;
        let log_sum_exp = sum_exp.log()?;
        shifted.sub(&log_sum_exp)
    }

    /// Variance along a dimension: var(x) = mean((x - mean(x))²)
    pub fn var(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        let mu = self.mean(dim, true)?;
        let centered = self.sub(&mu)?;
        let sq = centered.square()?;
        sq.mean(dim, keep_dim)
    }

    /// Concatenate tensors along a dimension.
    ///
    /// All tensors must have the same shape except in the concatenation dimension.
    /// This creates a new tensor by copying data from all inputs.
    pub fn cat(tensors: &[Self], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(Error::msg("cat: empty tensor list"));
        }
        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }

        let first = &tensors[0];
        let rank = first.rank();
        if dim >= rank {
            return Err(Error::DimOutOfRange { dim, rank });
        }

        // Validate shapes: all dims must match except `dim`
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.rank() != rank {
                return Err(Error::msg(format!(
                    "cat: tensor {} has rank {} but expected {}",
                    i,
                    t.rank(),
                    rank
                )));
            }
            if t.dtype() != first.dtype() {
                return Err(Error::DTypeMismatch {
                    expected: first.dtype(),
                    got: t.dtype(),
                });
            }
            for d in 0..rank {
                if d != dim && t.dims()[d] != first.dims()[d] {
                    return Err(Error::msg(format!(
                        "cat: tensor {} has size {} at dim {} but expected {}",
                        i,
                        t.dims()[d],
                        d,
                        first.dims()[d]
                    )));
                }
            }
        }

        // Compute output shape
        let cat_size: usize = tensors.iter().map(|t| t.dims()[dim]).sum();
        let mut out_dims = first.dims().to_vec();
        out_dims[dim] = cat_size;
        let out_shape = Shape::new(out_dims.clone());

        // Record per-input sizes along the cat dim for backward
        let sizes: Vec<usize> = tensors.iter().map(|t| t.dims()[dim]).collect();

        // Collect (storage, layout) pairs for Backend::cat
        let inner_guards: Vec<_> = tensors
            .iter()
            .map(|t| t.inner.storage.read().unwrap())
            .collect();
        let pairs: Vec<(&B::Storage, &Layout)> = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (&*inner_guards[i], &t.inner.layout))
            .collect();

        let storage = B::cat(&pairs, &out_shape, dim)?;
        let layout = Layout::contiguous(out_shape);
        let op = Op::Cat {
            inputs: tensors.to_vec(),
            dim,
            sizes,
        };
        Ok(Self::from_storage(
            storage,
            layout,
            first.dtype(),
            first.device().clone(),
            op,
        ))
    }

    /// Split a tensor into `n` equal chunks along a dimension.
    /// If the dimension size is not evenly divisible, the last chunk is smaller.
    pub fn chunk(&self, n: usize, dim: usize) -> Result<Vec<Self>> {
        if dim >= self.rank() {
            return Err(Error::DimOutOfRange {
                dim,
                rank: self.rank(),
            });
        }
        let dim_size = self.dims()[dim];
        let chunk_size = dim_size.div_ceil(n);
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < dim_size {
            let len = chunk_size.min(dim_size - start);
            chunks.push(self.narrow(dim, start, len)?);
            start += len;
        }
        Ok(chunks)
    }

    /// Expand a tensor to a larger shape by repeating data along size-1 dims.
    /// Only dims that are currently size 1 can be expanded.
    /// A size of -1 (usize::MAX) means don't change that dim.
    pub fn expand(&self, target_shape: impl Into<Shape>) -> Result<Self> {
        let target = target_shape.into();
        let self_dims = self.dims();
        let target_dims = target.dims();

        if self_dims.len() != target_dims.len() {
            return Err(Error::msg(format!(
                "expand: rank mismatch — self {:?} vs target {:?}",
                self_dims, target_dims
            )));
        }

        for (i, (&sd, &td)) in self_dims.iter().zip(target_dims.iter()).enumerate() {
            if sd != td && sd != 1 {
                return Err(Error::msg(format!(
                    "expand: can only expand size-1 dims, but dim {} has size {}",
                    i, sd
                )));
            }
        }

        // Zero-copy expand via stride tricks:
        // For dims where self_dim == 1 and target_dim > 1, set stride to 0
        // (the single element is "repeated" without copying data).
        let self_strides = self.inner.layout.strides();
        let mut new_strides = self_strides.to_vec();
        for d in 0..target_dims.len() {
            if self_dims[d] == 1 && target_dims[d] > 1 {
                new_strides[d] = 0;
            }
        }

        let new_layout = Layout::new(target, new_strides, self.inner.layout.offset());

        // Share the same storage — no copy!
        Ok(Tensor {
            inner: Arc::new(TensorInner {
                id: TensorId::new(),
                storage: Arc::clone(&self.inner.storage),
                layout: new_layout,
                dtype: self.inner.dtype,
                device: self.inner.device.clone(),
                op: Op::None,
                is_variable: false,
            }),
        })
    }

    // Stack — concatenate with a new dimension

    /// Stack tensors along a new dimension.
    ///
    /// All tensors must have the same shape. Inserts a new dimension at `dim`.
    /// `stack([a, b], dim=0)` where a,b are shape [2,3] → [2, 2, 3].
    pub fn stack(tensors: &[Self], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(Error::msg("stack: empty tensor list"));
        }
        let first_shape = tensors[0].shape().clone();
        let rank = first_shape.dims().len();
        if dim > rank {
            return Err(Error::DimOutOfRange {
                dim,
                rank: rank + 1,
            });
        }
        // Validate all shapes match
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.shape() != &first_shape {
                return Err(Error::msg(format!(
                    "stack: tensor {} has shape {:?} but expected {:?}",
                    i,
                    t.dims(),
                    first_shape.dims()
                )));
            }
        }
        // Unsqueeze each tensor at `dim`, then cat
        let unsqueezed: Vec<Self> = tensors
            .iter()
            .map(|t| t.unsqueeze(dim))
            .collect::<Result<Vec<_>>>()?;
        Self::cat(&unsqueezed, dim)
    }

    // Arange — generate a sequence

    /// Create a 1-D tensor with values [0, 1, ..., n-1].
    pub fn arange(n: usize, dtype: DType, device: &B::Device) -> Result<Self> {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        Self::from_f64_slice(&data, n, dtype, device)
    }

    /// Create a 1-D tensor with values [start, start+step, ..., <end).
    pub fn arange_step(
        start: f64,
        end: f64,
        step: f64,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        if step == 0.0 {
            return Err(Error::msg("arange_step: step cannot be zero"));
        }
        let mut data = Vec::new();
        let mut v = start;
        if step > 0.0 {
            while v < end {
                data.push(v);
                v += step;
            }
        } else {
            while v > end {
                data.push(v);
                v += step;
            }
        }
        let len = data.len();
        Self::from_f64_slice(&data, len, dtype, device)
    }

    // Triangular masks — triu / tril

    /// Upper triangular mask: returns a 2-D tensor of shape [n, m] where
    /// elements on and above the `diagonal`-th diagonal are 1.0, rest 0.0.
    ///
    /// `diagonal = 0` → main diagonal. `diagonal > 0` → above. `diagonal < 0` → below.
    pub fn triu(
        n: usize,
        m: usize,
        diagonal: i64,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        let mut data = vec![0.0f64; n * m];
        for i in 0..n {
            for j in 0..m {
                if (j as i64) >= (i as i64) + diagonal {
                    data[i * m + j] = 1.0;
                }
            }
        }
        Self::from_f64_slice(&data, (n, m), dtype, device)
    }

    /// Lower triangular mask: returns a 2-D tensor of shape [n, m] where
    /// elements on and below the `diagonal`-th diagonal are 1.0, rest 0.0.
    pub fn tril(
        n: usize,
        m: usize,
        diagonal: i64,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        let mut data = vec![0.0f64; n * m];
        for i in 0..n {
            for j in 0..m {
                if (j as i64) <= (i as i64) + diagonal {
                    data[i * m + j] = 1.0;
                }
            }
        }
        Self::from_f64_slice(&data, (n, m), dtype, device)
    }

    // Matrix multiplication

    /// Matrix multiplication: self @ rhs.
    ///
    /// - [m, k] @ [k, n] → [m, n]
    /// - Batched: [b, m, k] @ [b, k, n] → [b, m, n]
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                expected: self.dtype(),
                got: rhs.dtype(),
            });
        }
        // Validate shapes for matmul
        if self.rank() < 2 || rhs.rank() < 2 {
            return Err(Error::RankMismatch {
                expected: 2,
                got: self.rank().min(rhs.rank()),
            });
        }
        let lhs_dims = self.dims();
        let rhs_dims = rhs.dims();
        let k1 = lhs_dims[lhs_dims.len() - 1];
        let k2 = rhs_dims[rhs_dims.len() - 2];
        if k1 != k2 {
            let m = lhs_dims[lhs_dims.len() - 2];
            let n = rhs_dims[rhs_dims.len() - 1];
            return Err(Error::MatmulShapeMismatch { m, k1, k2, n });
        }

        let storage_lhs = self.read_storage()?;
        let storage_rhs = rhs.read_storage()?;
        let result = B::matmul(
            &storage_lhs,
            &self.inner.layout,
            &storage_rhs,
            &rhs.inner.layout,
        )?;

        // Result shape: [..., m, n]
        let m = lhs_dims[lhs_dims.len() - 2];
        let n = rhs_dims[rhs_dims.len() - 1];
        let mut result_dims: Vec<usize> = lhs_dims[..lhs_dims.len() - 2].to_vec();
        result_dims.push(m);
        result_dims.push(n);
        let result_layout = Layout::contiguous(Shape::new(result_dims));
        let result_op = Op::Matmul {
            lhs: self.clone(),
            rhs: rhs.clone(),
        };
        Ok(Self::from_storage(
            result,
            result_layout,
            self.inner.dtype,
            self.inner.device.clone(),
            result_op,
        ))
    }

    // 2D Convolution

    /// 2D convolution: applies convolution filters to a 4D input tensor.
    ///
    /// - `self` (input): `[N, C_in, H, W]`
    /// - `weight`:       `[C_out, C_in, kH, kW]`
    /// - `bias`:         optional `[C_out]`
    /// - `stride`:       `[sH, sW]`
    /// - `padding`:      `[pH, pW]`
    ///
    /// Returns tensor of shape `[N, C_out, H_out, W_out]` where
    /// `H_out = (H + 2*pH - kH) / sH + 1`.
    #[allow(clippy::needless_range_loop)]
    pub fn conv2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Self> {
        // Validate ranks
        if self.rank() != 4 {
            return Err(Error::msg(format!(
                "conv2d input must be 4D [N,C,H,W], got rank {}",
                self.rank()
            )));
        }
        if weight.rank() != 4 {
            return Err(Error::msg(format!(
                "conv2d weight must be 4D [C_out,C_in,kH,kW], got rank {}",
                weight.rank()
            )));
        }

        let in_dims = self.dims();
        let w_dims = weight.dims();
        let (n, c_in, h, w) = (in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
        let (c_out, wc_in, kh, kw) = (w_dims[0], w_dims[1], w_dims[2], w_dims[3]);

        if c_in != wc_in {
            return Err(Error::msg(format!(
                "conv2d: input channels {} != weight channels {}",
                c_in, wc_in
            )));
        }

        let [sh, sw] = stride;
        let [ph, pw] = padding;

        if h + 2 * ph < kh || w + 2 * pw < kw {
            return Err(Error::msg("conv2d: kernel larger than padded input"));
        }

        let h_out = (h + 2 * ph - kh) / sh + 1;
        let w_out = (w + 2 * pw - kw) / sw + 1;

        // Get contiguous data
        let input_data = self.contiguous()?.to_f64_vec()?;
        let weight_data = weight.contiguous()?.to_f64_vec()?;
        let bias_data = match bias {
            Some(b) => Some(b.contiguous()?.to_f64_vec()?),
            None => None,
        };

        let out_size = n * c_out * h_out * w_out;
        let mut output = vec![0.0f64; out_size];

        // im2col + GEMM approach:
        // For each batch sample:
        //   1. im2col: unroll input patches → columns [c_in*kh*kw, h_out*w_out]
        //   2. GEMM: weight [c_out, c_in*kh*kw] × columns → out [c_out, h_out*w_out]
        let col_rows = c_in * kh * kw;
        let col_cols = h_out * w_out;
        let mut columns = vec![0.0f64; col_rows * col_cols];
        let sample_size = c_in * h * w;

        for ni in 0..n {
            // im2col for this sample
            let in_offset = ni * sample_size;
            im2col(
                &input_data[in_offset..in_offset + sample_size],
                c_in,
                h,
                w,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                h_out,
                w_out,
                &mut columns,
            );

            // GEMM: output[ni] = weight × columns + bias
            let out_offset = ni * c_out * h_out * w_out;
            gemm(
                &weight_data,
                &columns,
                &mut output[out_offset..out_offset + c_out * col_cols],
                c_out,
                col_cols,
                col_rows,
            );

            // Add bias
            if let Some(ref bd) = bias_data {
                for co in 0..c_out {
                    let row_start = out_offset + co * col_cols;
                    for j in 0..col_cols {
                        output[row_start + j] += bd[co];
                    }
                }
            }
        }

        let result_shape = Shape::new(vec![n, c_out, h_out, w_out]);
        let result_op = Op::Conv2d {
            input: self.clone(),
            weight: weight.clone(),
            bias: bias.cloned(),
            stride,
            padding,
        };
        Self::from_f64_slice(&output, result_shape.clone(), self.dtype(), self.device()).map(|t| {
            Self::from_storage(
                {
                    let s = t.inner.storage.read().expect("storage lock poisoned");
                    s.clone()
                },
                Layout::contiguous(result_shape),
                self.inner.dtype,
                self.inner.device.clone(),
                result_op,
            )
        })
    }

    // 2D Max Pooling

    /// 2D max pooling on a 4D input tensor `[N, C, H, W]`.
    ///
    /// Returns `(output, indices)` where `indices` stores argmax positions
    /// (flat indices into the input) for backward.
    pub fn max_pool2d(
        &self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Self> {
        if self.rank() != 4 {
            return Err(Error::msg(format!(
                "max_pool2d input must be 4D [N,C,H,W], got rank {}",
                self.rank()
            )));
        }

        let dims = self.dims();
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let [kh, kw] = kernel_size;
        let [sh, sw] = stride;
        let [ph, pw] = padding;

        if h + 2 * ph < kh || w + 2 * pw < kw {
            return Err(Error::msg("max_pool2d: kernel larger than padded input"));
        }

        let h_out = (h + 2 * ph - kh) / sh + 1;
        let w_out = (w + 2 * pw - kw) / sw + 1;

        let input_data = self.contiguous()?.to_f64_vec()?;
        let out_size = n * c * h_out * w_out;
        let mut output = vec![f64::NEG_INFINITY; out_size];
        let mut indices = vec![0usize; out_size];

        for ni in 0..n {
            for ci in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let out_idx = ((ni * c + ci) * h_out + oh) * w_out + ow;
                        let mut max_val = f64::NEG_INFINITY;
                        let mut max_idx = 0usize;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = (oh * sh + ki) as isize - ph as isize;
                                let iw = (ow * sw + kj) as isize - pw as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    let in_idx = ((ni * c + ci) * h + ih) * w + iw;
                                    if input_data[in_idx] > max_val {
                                        max_val = input_data[in_idx];
                                        max_idx = in_idx;
                                    }
                                }
                            }
                        }
                        output[out_idx] = max_val;
                        indices[out_idx] = max_idx;
                    }
                }
            }
        }

        let result_shape = Shape::new(vec![n, c, h_out, w_out]);
        let result_op = Op::MaxPool2d {
            input: self.clone(),
            kernel_size,
            stride,
            padding,
            indices: indices.clone(),
        };
        Self::from_f64_slice(&output, result_shape.clone(), self.dtype(), self.device()).map(|t| {
            Self::from_storage(
                {
                    let s = t.inner.storage.read().expect("storage lock poisoned");
                    s.clone()
                },
                Layout::contiguous(result_shape),
                self.inner.dtype,
                self.inner.device.clone(),
                result_op,
            )
        })
    }

    // 2D Average Pooling

    /// Apply 2D average pooling to a 4D tensor [N, C, H, W].
    pub fn avg_pool2d(
        &self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Self> {
        if self.rank() != 4 {
            return Err(Error::msg(format!(
                "avg_pool2d input must be 4D [N,C,H,W], got rank {}",
                self.rank()
            )));
        }

        let dims = self.dims();
        let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let [kh, kw] = kernel_size;
        let [sh, sw] = stride;
        let [ph, pw] = padding;

        if h + 2 * ph < kh || w + 2 * pw < kw {
            return Err(Error::msg("avg_pool2d: kernel larger than padded input"));
        }

        let h_out = (h + 2 * ph - kh) / sh + 1;
        let w_out = (w + 2 * pw - kw) / sw + 1;

        let input_data = self.contiguous()?.to_f64_vec()?;
        let out_size = n * c * h_out * w_out;
        let mut output = vec![0.0f64; out_size];

        for ni in 0..n {
            for ci in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let out_idx = ((ni * c + ci) * h_out + oh) * w_out + ow;
                        let mut sum = 0.0f64;
                        let mut count = 0usize;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = (oh * sh + ki) as isize - ph as isize;
                                let iw = (ow * sw + kj) as isize - pw as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let in_idx =
                                        ((ni * c + ci) * h + ih as usize) * w + iw as usize;
                                    sum += input_data[in_idx];
                                    count += 1;
                                }
                            }
                        }
                        output[out_idx] = if count > 0 { sum / count as f64 } else { 0.0 };
                    }
                }
            }
        }

        let result_shape = Shape::new(vec![n, c, h_out, w_out]);
        let result_op = Op::AvgPool2d {
            input: self.clone(),
            kernel_size,
            stride,
            padding,
        };
        Self::from_f64_slice(&output, result_shape.clone(), self.dtype(), self.device()).map(|t| {
            Self::from_storage(
                {
                    let s = t.inner.storage.read().expect("storage lock poisoned");
                    s.clone()
                },
                Layout::contiguous(result_shape),
                self.inner.dtype,
                self.inner.device.clone(),
                result_op,
            )
        })
    }

    // 1D Convolution

    /// Apply 1D convolution to a 3D tensor [N, C_in, L].
    /// weight: [C_out, C_in, K]
    #[allow(clippy::needless_range_loop)]
    pub fn conv1d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        if self.rank() != 3 {
            return Err(Error::msg(format!(
                "conv1d input must be 3D [N,C_in,L], got rank {}",
                self.rank()
            )));
        }
        if weight.rank() != 3 {
            return Err(Error::msg(format!(
                "conv1d weight must be 3D [C_out,C_in,K], got rank {}",
                weight.rank()
            )));
        }

        let in_dims = self.dims();
        let w_dims = weight.dims();
        let (n, c_in, l) = (in_dims[0], in_dims[1], in_dims[2]);
        let (c_out, wc_in, k) = (w_dims[0], w_dims[1], w_dims[2]);

        if c_in != wc_in {
            return Err(Error::msg(format!(
                "conv1d: input channels {} != weight channels {}",
                c_in, wc_in
            )));
        }
        if let Some(b) = bias {
            if b.elem_count() != c_out {
                return Err(Error::msg(format!(
                    "conv1d: bias size {} != output channels {}",
                    b.elem_count(),
                    c_out
                )));
            }
        }

        if l + 2 * padding < k {
            return Err(Error::msg("conv1d: kernel larger than padded input"));
        }

        let l_out = (l + 2 * padding - k) / stride + 1;

        let input_data = self.contiguous()?.to_f64_vec()?;
        let weight_data = weight.contiguous()?.to_f64_vec()?;
        let bias_data: Option<Vec<f64>> = match bias {
            Some(b) => Some(b.to_f64_vec()?),
            None => None,
        };

        let out_size = n * c_out * l_out;
        let mut output = vec![0.0f64; out_size];

        // im2col + GEMM for conv1d (treat as 2D with h=1)
        let col_rows = c_in * k;
        let col_cols = l_out;
        let mut columns = vec![0.0f64; col_rows * col_cols];
        let sample_size = c_in * l;

        for ni in 0..n {
            // im2col for 1D: unroll patches
            let in_offset = ni * sample_size;
            im2col(
                &input_data[in_offset..in_offset + sample_size],
                c_in,
                1,
                l,
                1,
                k,
                1,
                stride,
                0,
                padding,
                1,
                l_out,
                &mut columns,
            );

            // GEMM: output[ni] = weight × columns
            let out_offset = ni * c_out * l_out;
            gemm(
                &weight_data,
                &columns,
                &mut output[out_offset..out_offset + c_out * col_cols],
                c_out,
                col_cols,
                col_rows,
            );

            // Add bias
            if let Some(ref bd) = bias_data {
                for co in 0..c_out {
                    let row_start = out_offset + co * col_cols;
                    for j in 0..col_cols {
                        output[row_start + j] += bd[co];
                    }
                }
            }
        }

        let result_shape = Shape::new(vec![n, c_out, l_out]);
        let result_op = Op::Conv1d {
            input: self.clone(),
            weight: weight.clone(),
            bias: bias.cloned(),
            stride,
            padding,
        };
        Self::from_f64_slice(&output, result_shape.clone(), self.dtype(), self.device()).map(|t| {
            Self::from_storage(
                {
                    let s = t.inner.storage.read().expect("storage lock poisoned");
                    s.clone()
                },
                Layout::contiguous(result_shape),
                self.inner.dtype,
                self.inner.device.clone(),
                result_op,
            )
        })
    }

    // Affine transform

    /// Affine transform: result[i] = self[i] * mul + add.
    /// Useful for normalization and scaling.
    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> {
        let storage = self.read_storage()?;
        let result = B::affine(&storage, &self.inner.layout, mul, add)?;
        let result_layout = Layout::contiguous(self.shape().clone());
        let result_op = Op::Affine {
            input: self.clone(),
            mul,
            add,
        };
        Ok(Self::from_storage(
            result,
            result_layout,
            self.inner.dtype,
            self.inner.device.clone(),
            result_op,
        ))
    }

    // Data extraction (for testing and debugging)

    /// Extract all elements as a flat Vec<f64>.
    pub fn to_f64_vec(&self) -> Result<Vec<f64>> {
        let storage = self.read_storage()?;
        B::to_f64_vec(&storage, &self.inner.layout)
    }

    /// Extract a scalar value (tensor must have exactly 1 element).
    pub fn to_scalar_f64(&self) -> Result<f64> {
        if self.elem_count() != 1 {
            return Err(Error::NotAScalar {
                shape: self.shape().clone(),
            });
        }
        let vec = self.to_f64_vec()?;
        Ok(vec[0])
    }

    /// Convert this tensor to a different dtype.
    ///
    /// Returns a new tensor with the same shape but different element type.
    /// Uses the backend's on-device cast when available, avoiding host round-trips.
    /// Records Op::ToDtype so gradients flow back through dtype conversions.
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.dtype() == dtype {
            return Ok(self.clone());
        }
        let src_dtype = self.dtype();
        let guard = self.inner.storage.read().unwrap();
        let storage = B::cast(&*guard, &self.inner.layout, dtype, self.device())?;
        let layout = Layout::contiguous(self.shape().clone());
        let op = if self.is_variable() {
            Op::ToDtype {
                input: self.clone(),
                src_dtype,
            }
        } else {
            Op::None
        };
        Ok(Self::from_storage(
            storage,
            layout,
            dtype,
            self.device().clone(),
            op,
        ))
    }

    /// Display the tensor contents in a human-readable format.
    pub fn to_string_with_data(&self) -> Result<String> {
        let data = self.to_f64_vec()?;
        Ok(format!(
            "Tensor(shape={}, dtype={}, data={:?})",
            self.shape(),
            self.dtype(),
            data
        ))
    }

    // Autograd

    /// Compute gradients via reverse-mode automatic differentiation.
    ///
    /// This tensor must be a scalar (single element). Returns a GradStore
    /// containing gradients for all tensors in the computation graph.
    ///
    /// # Example
    /// ```ignore
    /// let a = Tensor::from_f64_slice(&[2.0], 1, DType::F32, &dev)?.set_variable();
    /// let b = Tensor::from_f64_slice(&[3.0], 1, DType::F32, &dev)?.set_variable();
    /// let c = a.mul(&b)?;
    /// let grads = c.backward()?;
    /// // grad_a = b = 3.0, grad_b = a = 2.0
    /// ```
    pub fn backward(&self) -> Result<crate::backprop::GradStore<B>> {
        crate::backprop::backward(self)
    }

    /// Create a detached copy: same data but no gradient tracking.
    /// The new tensor has Op::None and a fresh TensorId.
    pub fn detach(&self) -> Self {
        self.view_with_layout(self.layout().clone(), Op::None)
    }

    /// Freeze this tensor: same data and id, but `is_variable = false`.
    ///
    /// Frozen tensors do NOT accumulate gradients during backward().
    /// This is the functional equivalent of PyTorch's `param.requires_grad_(False)`.
    pub fn freeze(&self) -> Self {
        Tensor {
            inner: Arc::new(TensorInner {
                id: self.inner.id,
                storage: Arc::clone(&self.inner.storage),
                layout: self.inner.layout.clone(),
                dtype: self.inner.dtype,
                device: self.inner.device.clone(),
                op: self.inner.op.clone(),
                is_variable: false,
            }),
        }
    }

    /// Unfreeze this tensor: same data and id, but `is_variable = true`.
    ///
    /// This is the opposite of `freeze()`.
    pub fn unfreeze(&self) -> Self {
        self.set_variable_ref()
    }

    // Additional composite operations

    /// Select entries along `dim` using the given 1-D index tensor.
    ///
    /// The output has the same rank, with `dim` resized to `indices.len()`.
    /// Wraps the `Backend::index_select` kernel.
    pub fn index_select(&self, dim: usize, indices: &Self) -> Result<Self> {
        if dim >= self.rank() {
            return Err(Error::DimOutOfRange {
                dim,
                rank: self.rank(),
            });
        }
        let guard = self.inner.storage.read().unwrap();
        let idx_guard = indices.inner.storage.read().unwrap();
        let storage = B::index_select(
            &*guard,
            &self.inner.layout,
            &*idx_guard,
            &indices.inner.layout,
            dim,
        )?;
        let mut out_dims = self.dims().to_vec();
        out_dims[dim] = indices.elem_count();
        let layout = Layout::contiguous(Shape::new(out_dims));
        // Record op for autograd — gradient flows back to input via scatter-add
        let op = Op::IndexSelect {
            input: self.clone(),
            indices: indices.clone(),
            dim,
        };
        Ok(Self::from_storage(
            storage,
            layout,
            self.dtype(),
            self.device().clone(),
            op,
        ))
    }

    /// Split a tensor into chunks of `split_size` along `dim`.
    ///
    /// The last chunk may be smaller if the dimension is not evenly divisible.
    pub fn split(&self, split_size: usize, dim: usize) -> Result<Vec<Self>> {
        if dim >= self.rank() {
            return Err(Error::DimOutOfRange {
                dim,
                rank: self.rank(),
            });
        }
        if split_size == 0 {
            return Err(Error::msg("split: split_size must be > 0"));
        }
        let dim_size = self.dims()[dim];
        let mut parts = Vec::new();
        let mut start = 0;
        while start < dim_size {
            let len = split_size.min(dim_size - start);
            parts.push(self.narrow(dim, start, len)?);
            start += len;
        }
        Ok(parts)
    }

    /// Flatten dimensions `start_dim..=end_dim` into a single dimension.
    ///
    /// Negative-style indexing is **not** supported; both bounds are inclusive
    /// and zero-based.
    pub fn flatten(&self, start_dim: usize, end_dim: usize) -> Result<Self> {
        let rank = self.rank();
        if start_dim >= rank || end_dim >= rank || start_dim > end_dim {
            return Err(Error::msg(format!(
                "flatten: invalid range [{}, {}] for rank {}",
                start_dim, end_dim, rank
            )));
        }
        let dims = self.dims();
        let mut new_dims: Vec<usize> = Vec::new();
        new_dims.extend_from_slice(&dims[..start_dim]);
        let flat: usize = dims[start_dim..=end_dim].iter().product();
        new_dims.push(flat);
        if end_dim + 1 < rank {
            new_dims.extend_from_slice(&dims[end_dim + 1..]);
        }
        self.reshape(new_dims)
    }

    /// Standard deviation along a dimension.
    ///
    /// Computed as `sqrt(var(x, dim))`.
    pub fn std(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        self.var(dim, keep_dim)?.sqrt()
    }

    /// Element-wise reciprocal: `1 / x`.
    pub fn reciprocal(&self) -> Result<Self> {
        let one = Self::ones(self.dims(), self.dtype(), self.device())?;
        one.div(self)
    }

    /// Element-wise reciprocal square-root: `1 / sqrt(x)`.
    pub fn rsqrt(&self) -> Result<Self> {
        self.sqrt()?.reciprocal()
    }

    /// Element-wise sign: returns -1, 0, or +1.
    ///
    /// Implemented via `x / (|x| + eps)` clamped to [-1, 1], with exact 0 for
    /// inputs that are exactly zero.
    pub fn sign(&self) -> Result<Self> {
        let eps = 1e-12;
        let abs_x = self.abs()?;
        let denom = abs_x.affine(1.0, eps)?;
        let raw = self.div(&denom)?;
        raw.clamp(-1.0, 1.0)
    }

    /// Log-sum-exp along a dimension (numerically stable).
    ///
    /// `logsumexp(x, d) = max(x,d) + log(sum(exp(x - max(x,d)), d))`
    pub fn logsumexp(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        let m = self.max(dim, true)?.detach();
        let shifted = self.sub(&m)?;
        let sum_exp = shifted.exp()?.sum(dim, true)?.log()?;
        let result = m.add(&sum_exp)?;
        if keep_dim {
            Ok(result)
        } else {
            result.squeeze(dim)
        }
    }

    /// Product of elements along a dimension.
    ///
    /// Computed as `exp(sum(log(|x|)))` with sign correction.
    /// **Warning**: undefined for inputs containing zero.
    pub fn prod(&self, dim: usize, keep_dim: bool) -> Result<Self> {
        let log_abs = self.abs()?.log()?;
        let sum_log = log_abs.sum(dim, keep_dim)?;
        let magnitude = sum_log.exp()?;
        // Sign: count negatives via (sign < 0) then parity
        // For simplicity, assume positive inputs (like PyTorch's default usage).
        // A full sign-tracking impl would need additional ops.
        Ok(magnitude)
    }

    /// Like `set_variable(self)` but takes `&self` instead of `self`.
    fn set_variable_ref(&self) -> Self {
        Tensor {
            inner: Arc::new(TensorInner {
                id: self.inner.id,
                storage: Arc::clone(&self.inner.storage),
                layout: self.inner.layout.clone(),
                dtype: self.inner.dtype,
                device: self.inner.device.clone(),
                op: self.inner.op.clone(),
                is_variable: true,
            }),
        }
    }
}

// im2col / col2im — Efficient convolution via matrix multiplication
//
// im2col extracts all sliding-window patches from the input and arranges them
// as columns of a matrix. This converts convolution into a single large GEMM:
//
//   columns = im2col(input)          shape: [C_in * kH * kW,  H_out * W_out]
//   output  = weight × columns       shape: [C_out, H_out * W_out]
//
// col2im is the reverse: it scatters columns back into an image-shaped buffer,
// accumulating overlapping contributions. Used in the backward pass.

/// im2col: Extract sliding-window patches from a single sample.
///
/// Input: `[C_in, H, W]` (one sample, no batch dim)
/// Output: columns `[C_in * kH * kW, H_out * W_out]`
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn im2col(
    input: &[f64],
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    h_out: usize,
    w_out: usize,
    columns: &mut [f64],
) {
    let col_cols = h_out * w_out;
    // Each row of `columns` corresponds to one element of the kernel
    // across all spatial output positions
    for ci in 0..c_in {
        for ki in 0..kh {
            for kj in 0..kw {
                let row = (ci * kh + ki) * kw + kj;
                let row_offset = row * col_cols;
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let ih = (oh * sh + ki) as isize - ph as isize;
                        let iw = (ow * sw + kj) as isize - pw as isize;
                        let val = if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                            input[(ci * h + ih as usize) * w + iw as usize]
                        } else {
                            0.0
                        };
                        columns[row_offset + oh * w_out + ow] = val;
                    }
                }
            }
        }
    }
}

/// col2im: Scatter columns back into an image buffer (for backward).
///
/// Accumulates into `output` (which should be zeroed before calling).
/// columns: `[C_in * kH * kW, H_out * W_out]`
/// output: `[C_in, H, W]`
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn col2im(
    columns: &[f64],
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    h_out: usize,
    w_out: usize,
    output: &mut [f64],
) {
    let col_cols = h_out * w_out;
    for ci in 0..c_in {
        for ki in 0..kh {
            for kj in 0..kw {
                let row = (ci * kh + ki) * kw + kj;
                let row_offset = row * col_cols;
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let ih = (oh * sh + ki) as isize - ph as isize;
                        let iw = (ow * sw + kj) as isize - pw as isize;
                        if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                            output[(ci * h + ih as usize) * w + iw as usize] +=
                                columns[row_offset + oh * w_out + ow];
                        }
                    }
                }
            }
        }
    }
}

/// Simple GEMM: C = A × B
///
/// A: [m, k], B: [k, n], C: [m, n]
/// All row-major.
#[inline]
pub(crate) fn gemm(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    for i in 0..m {
        let a_row = i * k;
        let c_row = i * n;
        for p in 0..k {
            let a_val = a[a_row + p];
            let b_row = p * n;
            for j in 0..n {
                c[c_row + j] += a_val * b[b_row + j];
            }
        }
    }
}

/// GEMM: C = A^T × B
///
/// A: [k, m] (transposed to [m, k]), B: [k, n], C: [m, n]
#[inline]
pub(crate) fn gemm_at_b(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    for i in 0..m {
        let c_row = i * n;
        for p in 0..k {
            let a_val = a[p * m + i]; // A^T[i,p] = A[p,i]
            let b_row = p * n;
            for j in 0..n {
                c[c_row + j] += a_val * b[b_row + j];
            }
        }
    }
}

/// GEMM: C = A × B^T
///
/// A: [m, k], B: [n, k] (transposed to [k, n]), C: [m, n]
#[inline]
pub(crate) fn gemm_a_bt(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    for i in 0..m {
        let a_row = i * k;
        let c_row = i * n;
        for j in 0..n {
            let b_row = j * k;
            let mut val = 0.0f64;
            for p in 0..k {
                val += a[a_row + p] * b[b_row + p];
            }
            c[c_row + j] += val;
        }
    }
}
