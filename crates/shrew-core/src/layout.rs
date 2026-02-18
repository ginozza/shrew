use crate::error::{Error, Result};
use crate::shape::Shape;

// Layout — Memory layout of a tensor (shape + strides + offset)
//
// The Layout decouples the *logical* shape of a tensor from how its data is
// arranged in memory. This is what makes operations like transpose, reshape,
// and slicing "free" (no data copy needed — just change the layout).
//
// KEY CONCEPTS:
//
// 1. **Strides**: How many elements to skip in the flat storage to move one
//    step along each dimension. A contiguous [2,3] matrix has strides [3,1]:
//    - row 0 starts at offset 0, row 1 starts at offset 3
//    - within a row, consecutive elements are 1 apart
//
// 2. **Transpose**: Just swap the strides (and shape). No data movement!
//    [2,3] with strides [3,1] → transpose → [3,2] with strides [1,3]
//    The same data, but now read column-major.
//
// 3. **Narrow/Slice**: Just adjust the offset and shape. Still same storage.
//    narrow(dim=1, start=1, len=2) on [2,3] →
//    Shape [2,2], offset += 1 * stride[1] = 1, same strides
//
// 4. **Contiguous check**: A tensor is contiguous if its strides match the
//    default row-major strides for its shape. Non-contiguous tensors need
//    to be made contiguous (data copy) before certain operations (like
//    passing to BLAS or CUDA kernels that expect contiguous memory).

/// Layout describes how a tensor's logical shape maps to flat storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    strides: Vec<usize>,
    /// Offset into the storage buffer where this tensor's data starts.
    /// Used by slicing/narrow operations to create views into existing storage.
    offset: usize,
}

impl Layout {
    /// Create a new contiguous layout for the given shape.
    /// Strides are computed as row-major (C-order).
    pub fn contiguous(shape: Shape) -> Self {
        let strides = shape.stride_contiguous();
        Layout {
            shape,
            strides,
            offset: 0,
        }
    }

    /// Create a layout with explicit strides and offset (for views).
    pub fn new(shape: Shape, strides: Vec<usize>, offset: usize) -> Self {
        Layout {
            shape,
            strides,
            offset,
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    /// Check if this layout is contiguous (row-major, no gaps).
    /// A tensor is contiguous if its strides equal the default strides
    /// for its shape AND offset is 0.
    pub fn is_contiguous(&self) -> bool {
        self.offset == 0 && self.strides == self.shape.stride_contiguous()
    }

    /// Transpose two dimensions. Returns a new layout with swapped shape/strides.
    /// This is a "free" operation — no data is copied.
    ///
    /// Example: [2, 3, 4] transpose(0, 2) → [4, 3, 2]
    ///          strides [12, 4, 1]         → [1, 4, 12]
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Layout> {
        let rank = self.rank();
        if dim0 >= rank || dim1 >= rank {
            return Err(Error::DimOutOfRange {
                dim: dim0.max(dim1),
                rank,
            });
        }
        let mut new_dims = self.shape.dims().to_vec();
        let mut new_strides = self.strides.clone();
        new_dims.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);
        Ok(Layout::new(Shape::new(new_dims), new_strides, self.offset))
    }

    /// Narrow (slice) along a dimension. Returns a new layout that is a view
    /// into the same storage with adjusted shape and offset.
    ///
    /// Example: tensor of shape [4, 6], narrow(dim=1, start=2, len=3)
    /// → shape [4, 3], offset += 2 * stride[1]
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Layout> {
        let rank = self.rank();
        if dim >= rank {
            return Err(Error::DimOutOfRange { dim, rank });
        }
        let dim_size = self.shape.dims()[dim];
        if start + len > dim_size {
            return Err(Error::NarrowOutOfBounds {
                dim,
                start,
                len,
                dim_size,
            });
        }
        let mut new_dims = self.shape.dims().to_vec();
        new_dims[dim] = len;
        let new_offset = self.offset + start * self.strides[dim];
        Ok(Layout::new(
            Shape::new(new_dims),
            self.strides.clone(),
            new_offset,
        ))
    }

    /// Compute the flat index into storage for a given multi-dimensional index.
    /// This is the core formula: flat_index = offset + sum(index[i] * stride[i])
    pub fn flat_index(&self, index: &[usize]) -> usize {
        let mut flat = self.offset;
        for (i, &idx) in index.iter().enumerate() {
            flat += idx * self.strides[i];
        }
        flat
    }

    /// Iterator over all flat indices of this layout, in logical order.
    /// This handles non-contiguous layouts correctly by walking through
    /// multi-dimensional indices and converting via strides.
    pub fn strided_indices(&self) -> StridedIter {
        StridedIter::new(self)
    }
}

// StridedIter — Iterates over flat storage indices respecting strides
//
// This iterator is essential for non-contiguous tensors. When a tensor has
// been transposed or sliced, the data in memory is no longer sequential.
// StridedIter walks through the logical elements in order and produces
// the actual storage index for each one.
//
// For a contiguous tensor, this just counts 0, 1, 2, 3, ...
// For a transposed tensor, it jumps around in memory following the strides.

/// Iterator that yields flat storage indices for each element of a Layout.
pub struct StridedIter {
    /// Current multi-dimensional index (e.g., [0, 0, 0]).
    current: Vec<usize>,
    /// The shape dimensions.
    dims: Vec<usize>,
    /// The strides for each dimension.
    strides: Vec<usize>,
    /// Base offset into storage.
    offset: usize,
    /// Total elements remaining.
    remaining: usize,
    /// Whether we've started yet.
    started: bool,
}

impl StridedIter {
    fn new(layout: &Layout) -> Self {
        let rank = layout.rank();
        StridedIter {
            current: vec![0; rank],
            dims: layout.dims().to_vec(),
            strides: layout.strides().to_vec(),
            offset: layout.offset(),
            remaining: layout.elem_count(),
            started: false,
        }
    }

    /// Compute flat index from current multi-dim index.
    fn flat_index(&self) -> usize {
        let mut idx = self.offset;
        for i in 0..self.current.len() {
            idx += self.current[i] * self.strides[i];
        }
        idx
    }

    /// Advance the multi-dimensional index by one (rightmost dimension first).
    fn advance(&mut self) {
        let rank = self.dims.len();
        for i in (0..rank).rev() {
            self.current[i] += 1;
            if self.current[i] < self.dims[i] {
                return;
            }
            self.current[i] = 0;
        }
    }
}

impl Iterator for StridedIter {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        if self.started {
            self.advance();
        }
        self.started = true;
        self.remaining -= 1;
        Some(self.flat_index())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for StridedIter {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;

    #[test]
    fn test_contiguous_layout() {
        let layout = Layout::contiguous(Shape::from((2, 3)));
        assert!(layout.is_contiguous());
        assert_eq!(layout.strides(), &[3, 1]);
        assert_eq!(layout.offset(), 0);
    }

    #[test]
    fn test_contiguous_indices() {
        // For [2, 3] contiguous, indices should be 0,1,2,3,4,5
        let layout = Layout::contiguous(Shape::from((2, 3)));
        let indices: Vec<usize> = layout.strided_indices().collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_transpose_layout() {
        let layout = Layout::contiguous(Shape::from((2, 3)));
        let transposed = layout.transpose(0, 1).unwrap();
        // Shape becomes [3, 2], strides become [1, 3]
        assert_eq!(transposed.dims(), &[3, 2]);
        assert_eq!(transposed.strides(), &[1, 3]);
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn test_transpose_indices() {
        // Original [2,3]:
        //   [[0, 1, 2],
        //    [3, 4, 5]]
        //
        // Transposed [3,2] should read column-major:
        //   [[0, 3],
        //    [1, 4],
        //    [2, 5]]
        //
        // So the flat indices in row-order of transposed are: 0, 3, 1, 4, 2, 5
        let layout = Layout::contiguous(Shape::from((2, 3)));
        let transposed = layout.transpose(0, 1).unwrap();
        let indices: Vec<usize> = transposed.strided_indices().collect();
        assert_eq!(indices, vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn test_narrow() {
        // [4, 6] narrow(dim=1, start=2, len=3) → [4, 3] with offset=2
        let layout = Layout::contiguous(Shape::from((4, 6)));
        let narrowed = layout.narrow(1, 2, 3).unwrap();
        assert_eq!(narrowed.dims(), &[4, 3]);
        assert_eq!(narrowed.offset(), 2);
        assert_eq!(narrowed.strides(), &[6, 1]); // strides unchanged
    }

    #[test]
    fn test_narrow_out_of_bounds() {
        let layout = Layout::contiguous(Shape::from((4, 6)));
        assert!(layout.narrow(1, 5, 3).is_err()); // 5+3 = 8 > 6
    }

    #[test]
    fn test_flat_index() {
        let layout = Layout::contiguous(Shape::from((2, 3, 4)));
        // Element at [1, 2, 3]: 1*12 + 2*4 + 3*1 = 23
        assert_eq!(layout.flat_index(&[1, 2, 3]), 23);
        // Element at [0, 0, 0]: 0
        assert_eq!(layout.flat_index(&[0, 0, 0]), 0);
    }
}
