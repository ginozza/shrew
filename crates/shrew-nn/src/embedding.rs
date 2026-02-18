// Embedding — Lookup table for discrete tokens
//
// An embedding layer maps integer indices to dense vectors. It's the standard
// way to handle categorical data (words, tokens, item IDs) in neural networks.
//
// Think of it as a learnable lookup table:
//   embedding[token_id] → vector of size embedding_dim
//
// For NLP: vocabulary of 50000 words → embedding(50000, 768) gives each word
// a 768-dimensional vector representation that the network learns during training.
//
// IMPLEMENTATION:
//
// The embedding table is a [num_embeddings, embedding_dim] matrix.
// Forward pass: given input indices [batch, seq_len], output is
// [batch, seq_len, embedding_dim] by looking up each index.
//
// For now we implement this as a gather operation using to_f64_vec and
// manual indexing. A more efficient index_select backend op can be added later.

use shrew_core::backend::Backend;
use shrew_core::dtype::DType;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use crate::module::Module;

/// A learnable lookup table mapping integer indices to dense vectors.
///
/// # Examples
/// ```ignore
/// let emb = Embedding::<CpuBackend>::new(1000, 128, DType::F32, &dev)?;
/// // Input: token indices [batch=2, seq_len=5]
/// let tokens = CpuTensor::from_f64_slice(&indices, (2, 5), DType::I64, &dev)?;
/// let vectors = emb.forward(&tokens)?; // [2, 5, 128]
/// ```
pub struct Embedding<B: Backend> {
    /// The embedding table: [num_embeddings, embedding_dim]
    weight: Tensor<B>,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl<B: Backend> Embedding<B> {
    /// Create a new Embedding layer with normally-distributed random weights.
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        // Initialize from N(0, 1) — standard for embeddings
        let weight =
            Tensor::<B>::randn((num_embeddings, embedding_dim), dtype, device)?.set_variable();
        Ok(Embedding {
            weight,
            num_embeddings,
            embedding_dim,
        })
    }

    /// Create from an existing weight matrix.
    pub fn from_tensor(weight: Tensor<B>) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(shrew_core::Error::msg(format!(
                "Embedding weight must be 2D, got shape {:?}",
                dims
            )));
        }
        Ok(Embedding {
            num_embeddings: dims[0],
            embedding_dim: dims[1],
            weight: weight.set_variable(),
        })
    }

    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    pub fn weight(&self) -> &Tensor<B> {
        &self.weight
    }
}

impl<B: Backend> Module<B> for Embedding<B> {
    /// Look up embeddings for the given indices.
    ///
    /// Input: integer tensor of any shape [...]
    /// Output: tensor of shape [..., embedding_dim]
    ///
    /// Uses `index_select` to gather rows from the weight matrix directly
    /// on-device (no host round-trip). For autograd, we record the operation
    /// so gradients can flow back through the embedding table.
    fn forward(&self, indices: &Tensor<B>) -> Result<Tensor<B>> {
        let input_dims = indices.dims().to_vec();
        let num_indices = indices.elem_count();

        // Flatten indices to 1D, ensure U32 for index_select
        let flat_idx = indices
            .reshape(shrew_core::Shape::new(vec![num_indices]))?
            .to_dtype(shrew_core::dtype::DType::U32)?;

        // index_select on dim=0: weight[flat_idx] → [num_indices, embedding_dim]
        let flat_result = self.weight.index_select(0, &flat_idx)?;

        // Reshape to [..., embedding_dim]
        let mut out_dims = input_dims;
        out_dims.push(self.embedding_dim);
        flat_result.reshape(shrew_core::Shape::new(out_dims))
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor<B>)> {
        vec![("weight".to_string(), self.weight.clone())]
    }
}
