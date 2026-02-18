use std::fmt;

// Shape — N-dimensional shape representation
//
// A Shape describes the size of each dimension of a tensor.
// For example:
//   - Scalar: Shape([])          — 0 dimensions, 1 element
//   - Vector: Shape([5])         — 1 dimension, 5 elements
//   - Matrix: Shape([3, 4])      — 2 dimensions, 12 elements
//   - Batch:  Shape([2, 3, 4])   — 3 dimensions, 24 elements
//
// The shape is fundamental because it determines:
//   1. How many elements are in the tensor (product of all dims)
//   2. The default (contiguous/row-major) strides for memory layout
//   3. Whether two tensors are compatible for operations (broadcasting rules)

/// N-dimensional shape of a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Create a new shape from a vector of dimension sizes.
    pub fn new(dims: Vec<usize>) -> Self {
        Shape(dims)
    }

    /// The dimension sizes as a slice.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Number of dimensions (0 for scalar, 1 for vector, 2 for matrix, etc.).
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements (product of all dimensions).
    /// A scalar shape [] has 1 element.
    pub fn elem_count(&self) -> usize {
        self.0.iter().product::<usize>().max(1)
    }

    /// Compute the contiguous (row-major / C-order) strides for this shape.
    ///
    /// For shape [2, 3, 4], strides are [12, 4, 1]:
    ///   - Moving 1 step in dim 0 jumps 12 elements (3*4)
    ///   - Moving 1 step in dim 1 jumps 4 elements
    ///   - Moving 1 step in dim 2 jumps 1 element
    ///
    /// This is how row-major memory works: the last dimension is contiguous.
    pub fn stride_contiguous(&self) -> Vec<usize> {
        let mut strides = vec![0usize; self.rank()];
        if self.rank() > 0 {
            strides[self.rank() - 1] = 1;
            for i in (0..self.rank() - 1).rev() {
                strides[i] = strides[i + 1] * self.0[i + 1];
            }
        }
        strides
    }

    /// Size of a specific dimension.
    pub fn dim(&self, d: usize) -> crate::Result<usize> {
        self.0.get(d).copied().ok_or(crate::Error::DimOutOfRange {
            dim: d,
            rank: self.rank(),
        })
    }

    // Broadcasting

    /// Compute the broadcast output shape from two input shapes.
    ///
    /// NumPy-style broadcasting rules:
    ///   1. Align shapes from the right (trailing dimensions).
    ///   2. Dimensions are compatible if they are equal or one of them is 1.
    ///   3. Missing leading dimensions are treated as 1.
    ///
    /// Examples:
    ///   [3, 4] and [4]     → [3, 4]   (expand [4] to [1, 4] then broadcast dim 0)
    ///   [2, 1] and [1, 3]  → [2, 3]
    ///   [5, 3, 1] and [3, 4] → [5, 3, 4]
    ///   [3] and [4]        → Error (3 ≠ 4 and neither is 1)
    pub fn broadcast_shape(lhs: &Shape, rhs: &Shape) -> crate::Result<Shape> {
        let l = lhs.dims();
        let r = rhs.dims();
        let max_rank = l.len().max(r.len());
        let mut result = Vec::with_capacity(max_rank);

        for i in 0..max_rank {
            // Index from the right: l.len()-1-i walks backwards. If i >= len, treat as 1.
            let ld = if i < l.len() { l[l.len() - 1 - i] } else { 1 };
            let rd = if i < r.len() { r[r.len() - 1 - i] } else { 1 };

            if ld == rd {
                result.push(ld);
            } else if ld == 1 {
                result.push(rd);
            } else if rd == 1 {
                result.push(ld);
            } else {
                return Err(crate::Error::msg(format!(
                    "shapes {:?} and {:?} are not broadcast-compatible (dim {} from right: {} vs {})",
                    l, r, i, ld, rd
                )));
            }
        }

        result.reverse(); // We built it from the right
        Ok(Shape::new(result))
    }

    /// Return the broadcast strides for this shape to match a target broadcast shape.
    ///
    /// For each dimension where self.dim[i] == 1 and target.dim[i] > 1,
    /// the stride is set to 0 (repeating the single element).
    /// For missing leading dimensions (self has fewer dims), stride is also 0.
    pub fn broadcast_strides(&self, target: &Shape) -> Vec<usize> {
        let self_dims = self.dims();
        let target_dims = target.dims();
        let self_strides = self.stride_contiguous();

        let mut result = vec![0usize; target_dims.len()];
        let offset = target_dims.len() - self_dims.len();

        for i in 0..self_dims.len() {
            if self_dims[i] == target_dims[i + offset] {
                result[i + offset] = self_strides[i];
            } else {
                // self_dims[i] must be 1 → stride 0 (broadcast)
                result[i + offset] = 0;
            }
        }
        // Leading dimensions (offset region) are already 0 (broadcast)
        result
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

// Convenient From implementations
// These let you write: Shape::from((3, 4)) instead of Shape::new(vec![3, 4])

impl From<()> for Shape {
    /// Scalar shape (0 dimensions).
    fn from(_: ()) -> Self {
        Shape(vec![])
    }
}

impl From<usize> for Shape {
    /// 1-D shape.
    fn from(d: usize) -> Self {
        Shape(vec![d])
    }
}

impl From<(usize,)> for Shape {
    fn from((d0,): (usize,)) -> Self {
        Shape(vec![d0])
    }
}

impl From<(usize, usize)> for Shape {
    fn from((d0, d1): (usize, usize)) -> Self {
        Shape(vec![d0, d1])
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from((d0, d1, d2): (usize, usize, usize)) -> Self {
        Shape(vec![d0, d1, d2])
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from((d0, d1, d2, d3): (usize, usize, usize, usize)) -> Self {
        Shape(vec![d0, d1, d2, d3])
    }
}

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        Shape(v)
    }
}

impl From<&[usize]> for Shape {
    fn from(s: &[usize]) -> Self {
        Shape(s.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_shape() {
        let s = Shape::from(());
        assert_eq!(s.rank(), 0);
        assert_eq!(s.elem_count(), 1);
        assert_eq!(s.stride_contiguous(), vec![]);
    }

    #[test]
    fn test_vector_shape() {
        let s = Shape::from(5);
        assert_eq!(s.rank(), 1);
        assert_eq!(s.elem_count(), 5);
        assert_eq!(s.stride_contiguous(), vec![1]);
    }

    #[test]
    fn test_matrix_shape() {
        let s = Shape::from((3, 4));
        assert_eq!(s.rank(), 2);
        assert_eq!(s.elem_count(), 12);
        // Row-major: stride for dim0 = 4, stride for dim1 = 1
        assert_eq!(s.stride_contiguous(), vec![4, 1]);
    }

    #[test]
    fn test_3d_strides() {
        let s = Shape::from((2, 3, 4));
        // [2,3,4]: strides = [3*4, 4, 1] = [12, 4, 1]
        assert_eq!(s.stride_contiguous(), vec![12, 4, 1]);
        assert_eq!(s.elem_count(), 24);
    }

    #[test]
    fn test_display() {
        let s = Shape::from((3, 4));
        assert_eq!(format!("{}", s), "[3, 4]");
    }
}
