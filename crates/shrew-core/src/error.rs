use crate::shape::Shape;

/// All errors that can occur within Shrew.
///
/// This enum captures every failure mode: shape mismatches, dtype mismatches,
/// device mismatches, out-of-bounds indexing, and backend-specific errors.
/// Using a single error type across the library simplifies error propagation.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Shape mismatch between two tensors (e.g., trying to add [2,3] + [4,5]).
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: Shape, got: Shape },

    /// Operation requires a specific rank (number of dimensions).
    #[error("rank mismatch: expected rank {expected}, got {got}")]
    RankMismatch { expected: usize, got: usize },

    /// DType mismatch between tensors in a binary operation.
    #[error("dtype mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch {
        expected: crate::DType,
        got: crate::DType,
    },

    /// Dimension index out of range for the tensor's rank.
    #[error("dimension out of range: dim {dim} for tensor with {rank} dimensions")]
    DimOutOfRange { dim: usize, rank: usize },

    /// Narrow/slice operation out of bounds.
    #[error("narrow out of bounds: dim {dim}, start {start}, len {len}, dim_size {dim_size}")]
    NarrowOutOfBounds {
        dim: usize,
        start: usize,
        len: usize,
        dim_size: usize,
    },

    /// Tried to access a scalar from a non-scalar tensor.
    #[error("not a scalar: tensor has shape {shape}")]
    NotAScalar { shape: Shape },

    /// Element count mismatch when creating from a vec.
    #[error("element count mismatch: shape {shape} requires {expected} elements, got {got}")]
    ElementCountMismatch {
        shape: Shape,
        expected: usize,
        got: usize,
    },

    /// Matrix multiplication dimension mismatch.
    #[error("matmul shape mismatch: [{m}x{k1}] @ [{k2}x{n}] — inner dims must match")]
    MatmulShapeMismatch {
        m: usize,
        k1: usize,
        k2: usize,
        n: usize,
    },

    /// Cannot reshape because element counts differ.
    #[error(
        "cannot reshape: source has {src} elements, target shape {dst_shape} has {dst} elements"
    )]
    ReshapeElementMismatch {
        src: usize,
        dst: usize,
        dst_shape: Shape,
    },

    /// Generic message for cases not covered above.
    #[error("{0}")]
    Msg(String),
}

impl Error {
    /// Create an error from any string message.
    pub fn msg(s: impl Into<String>) -> Self {
        Error::Msg(s.into())
    }
}

/// Convenience Result type used throughout Shrew.
pub type Result<T> = std::result::Result<T, Error>;

/// Macro for early return with a formatted error message.
/// Usage: `bail!("something went wrong: {}", detail)`
#[macro_export]
macro_rules! bail {
    ($($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($($arg)*)))
    };
}
