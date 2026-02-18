use std::fmt;

// DType — Supported numeric data types
//
// Every tensor has a DType that determines its element size and numeric
// behavior. We support the most common types for deep learning:
//
//   F16  — 16-bit IEEE half float, for mixed-precision training
//   BF16 — 16-bit brain float, for mixed-precision training
//   F32  — 32-bit float, the default workhorse
//   F64  — 64-bit float, for high-precision work
//   U8   — unsigned byte, for image data and boolean masks
//   U32  — unsigned 32-bit int, for indices
//   I64  — signed 64-bit int, for labels/indices (PyTorch convention)

/// Enum of all supported element data types.
///
/// This is stored inside every tensor so we can dispatch operations
/// to the correct typed implementation at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    BF16,
    F32,
    F64,
    U8,
    U32,
    I64,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
            DType::U8 => 1,
            DType::U32 => 4,
            DType::I64 => 8,
        }
    }

    /// Whether this dtype is a floating-point type (needed for gradient tracking).
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    /// Whether this is a half-precision type (F16 or BF16).
    pub fn is_half(&self) -> bool {
        matches!(self, DType::F16 | DType::BF16)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::U8 => "u8",
            DType::U32 => "u32",
            DType::I64 => "i64",
        };
        write!(f, "{}", s)
    }
}

// WithDType — Trait that connects Rust types to DType enum
//
// This trait is the bridge between Rust's type system and our runtime DType.
// By implementing it for f32, f64, etc., we can write generic functions like:
//
//   fn zeros<T: WithDType>(shape: Shape) -> Tensor { ... }
//
// and have the DType automatically determined from T.

/// Trait implemented by Rust types that can be stored in a tensor.
///
/// Provides the mapping between the concrete Rust type and the DType enum,
/// plus conversions to/from f64 for numeric operations.
pub trait WithDType: Copy + Send + Sync + 'static + num_traits::NumCast + std::fmt::Debug {
    /// The corresponding DType enum variant.
    const DTYPE: DType;

    /// Convert this value to f64 (for generic numeric code).
    fn to_f64(self) -> f64;

    /// Create a value of this type from f64.
    fn from_f64(v: f64) -> Self;

    /// The zero value.
    fn zero() -> Self {
        Self::from_f64(0.0)
    }

    /// The one value.
    fn one() -> Self {
        Self::from_f64(1.0)
    }
}

impl WithDType for f32 {
    const DTYPE: DType = DType::F32;
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl WithDType for f64 {
    const DTYPE: DType = DType::F64;
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(v: f64) -> Self {
        v
    }
}

impl WithDType for half::f16 {
    const DTYPE: DType = DType::F16;
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }
    fn from_f64(v: f64) -> Self {
        half::f16::from_f64(v)
    }
}

impl WithDType for half::bf16 {
    const DTYPE: DType = DType::BF16;
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }
    fn from_f64(v: f64) -> Self {
        half::bf16::from_f64(v)
    }
}

impl WithDType for u8 {
    const DTYPE: DType = DType::U8;
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as u8
    }
}

impl WithDType for u32 {
    const DTYPE: DType = DType::U32;
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as u32
    }
}

impl WithDType for i64 {
    const DTYPE: DType = DType::I64;
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert_eq!(DType::BF16.size_in_bytes(), 2);
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F64.size_in_bytes(), 8);
        assert_eq!(DType::U8.size_in_bytes(), 1);
    }

    #[test]
    fn test_dtype_is_half() {
        assert!(DType::F16.is_half());
        assert!(DType::BF16.is_half());
        assert!(!DType::F32.is_half());
        assert!(!DType::F64.is_half());
    }

    #[test]
    fn test_with_dtype_f32() {
        assert_eq!(f32::DTYPE, DType::F32);
        assert_eq!(f32::from_f64(3.14).to_f64(), 3.140000104904175); // f32 precision
    }

    #[test]
    fn test_with_dtype_roundtrip() {
        let v: f64 = 42.0;
        assert_eq!(f64::from_f64(v).to_f64(), v);
        assert_eq!(i64::from_f64(v).to_f64(), v);
        assert_eq!(u32::from_f64(v).to_f64(), v);
    }
}
