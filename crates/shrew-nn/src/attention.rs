// Multi-Head Attention — The core mechanism of the Transformer
//
// Multi-Head Attention (MHA) allows the model to jointly attend to information
// from different representation subspaces at different positions.
//
// INTUITION:
//
// Imagine you're reading a sentence: "The cat sat on the mat."
// For the word "sat", attention answers: "Which other words are relevant?"
//   - The query ("sat") asks: "What am I looking for?"
//   - The keys (all words) answer: "What do I contain?"
//   - The values (all words) say: "Here's what I provide."
//
// The attention score between query q and key k is: score = q · k / √d_k
// High score = this key is relevant to this query.
//
// MULTIPLE HEADS:
//
// Instead of one big attention, we split into h heads:
//   1. Project Q, K, V into h smaller subspaces (d_head = d_model / h)
//   2. Each head computes attention independently
//   3. Concatenate results and project back to d_model
//
// This lets different heads learn different types of relationships:
//   - Head 1 might learn syntactic dependencies
//   - Head 2 might learn semantic similarity
//   - Head 3 might learn positional patterns
//
// MATHEMATICS:
//
//   Input: x of shape [batch, seq_len, d_model]
//
//   1. Q = x @ W_Q    [batch, seq, d_model]
//      K = x @ W_K    [batch, seq, d_model]
//      V = x @ W_V    [batch, seq, d_model]
//
//   2. Reshape to heads: [batch, seq, h, d_head] → [batch, h, seq, d_head]
//
//   3. Attention per head:
//      scores = Q @ K^T / √d_head     [batch, h, seq, seq]
//      weights = softmax(scores, dim=-1)
//      out = weights @ V               [batch, h, seq, d_head]
//
//   4. Concatenate heads: [batch, seq, h * d_head] = [batch, seq, d_model]
//
//   5. Output projection: out @ W_O    [batch, seq, d_model]
//
// CAUSAL MASK (for autoregressive models like GPT):
//
//   To prevent attending to future tokens, we add a mask before softmax:
//   scores[i, j] = -∞  if j > i  (position j is after position i)
//   This makes softmax(scores[i, j]) = 0 for future positions.

use shrew_core::backend::Backend;
use shrew_core::dtype::DType;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use crate::linear::Linear;
use crate::module::Module;

/// Multi-Head Self-Attention module.
///
/// # Examples
/// ```ignore
/// let attn = MultiHeadAttention::<CpuBackend>::new(512, 8, DType::F64, &dev)?;
/// let x = CpuTensor::rand((2, 10, 512), DType::F64, &dev)?;
/// let y = attn.forward(&x)?; // [2, 10, 512]
/// ```
pub struct MultiHeadAttention<B: Backend> {
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head: d_head = d_model / num_heads
    head_dim: usize,
    /// Total model dimension
    d_model: usize,
    /// Query projection: [d_model, d_model]
    w_q: Linear<B>,
    /// Key projection: [d_model, d_model]
    w_k: Linear<B>,
    /// Value projection: [d_model, d_model]
    w_v: Linear<B>,
    /// Output projection: [d_model, d_model]
    w_o: Linear<B>,
    /// Scaling factor: 1/√d_head
    scale: f64,
    /// Whether to apply causal mask (for autoregressive models)
    causal: bool,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Create a new Multi-Head Attention module.
    ///
    /// # Arguments
    /// - `d_model`: total model dimension (must be divisible by num_heads)
    /// - `num_heads`: number of attention heads
    /// - `dtype`: data type for parameters
    /// - `device`: device to create parameters on
    pub fn new(d_model: usize, num_heads: usize, dtype: DType, device: &B::Device) -> Result<Self> {
        if !d_model.is_multiple_of(num_heads) {
            return Err(shrew_core::Error::msg(format!(
                "d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            )));
        }
        let head_dim = d_model / num_heads;

        let w_q = Linear::new(d_model, d_model, false, dtype, device)?;
        let w_k = Linear::new(d_model, d_model, false, dtype, device)?;
        let w_v = Linear::new(d_model, d_model, false, dtype, device)?;
        let w_o = Linear::new(d_model, d_model, false, dtype, device)?;

        Ok(MultiHeadAttention {
            num_heads,
            head_dim,
            d_model,
            w_q,
            w_k,
            w_v,
            w_o,
            scale: 1.0 / (head_dim as f64).sqrt(),
            causal: false,
        })
    }

    /// Enable causal (autoregressive) masking.
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Reshape [batch, seq, d_model] → [batch, seq, num_heads, head_dim]
    /// then transpose to [batch, num_heads, seq, head_dim]
    fn reshape_to_heads(&self, x: &Tensor<B>, batch: usize, seq: usize) -> Result<Tensor<B>> {
        // [batch, seq, d_model] → [batch, seq, num_heads, head_dim]
        let reshaped = x.reshape((batch, seq, self.num_heads, self.head_dim))?;
        // Transpose dims 1 and 2: [batch, num_heads, seq, head_dim]
        reshaped.transpose(1, 2)?.contiguous()
    }

    /// Inverse of reshape_to_heads:
    /// [batch, num_heads, seq, head_dim] → [batch, seq, d_model]
    fn reshape_from_heads(&self, x: &Tensor<B>, batch: usize, seq: usize) -> Result<Tensor<B>> {
        // [batch, num_heads, seq, head_dim] → [batch, seq, num_heads, head_dim]
        let transposed = x.transpose(1, 2)?.contiguous()?;
        // [batch, seq, num_heads * head_dim] = [batch, seq, d_model]
        transposed.reshape((batch, seq, self.d_model))
    }

    /// Create a causal mask: upper-triangular matrix of -infinity.
    /// Shape: [seq, seq] where mask[i][j] = -1e9 if j > i, else 0
    fn create_causal_mask(
        &self,
        seq_len: usize,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Tensor<B>> {
        let mut mask_data = vec![0.0f64; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = -1e9; // Large negative ≈ -infinity
            }
        }
        Tensor::<B>::from_f64_slice(&mask_data, (seq_len, seq_len), dtype, device)
    }
}

impl<B: Backend> Module<B> for MultiHeadAttention<B> {
    /// Forward pass: self-attention on input x.
    ///
    /// Input:  [batch, seq_len, d_model]
    /// Output: [batch, seq_len, d_model]
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let dims = x.dims();
        if dims.len() != 3 {
            return Err(shrew_core::Error::msg(format!(
                "MultiHeadAttention expects 3D input [batch, seq, d_model], got {:?}",
                dims
            )));
        }
        let batch = dims[0];
        let seq = dims[1];

        // Step 1: Project to Q, K, V
        // Reshape x from [batch, seq, d_model] to [batch*seq, d_model] for Linear
        let x_2d = x.reshape((batch * seq, self.d_model))?;
        let q_2d = self.w_q.forward(&x_2d)?; // [batch*seq, d_model]
        let k_2d = self.w_k.forward(&x_2d)?;
        let v_2d = self.w_v.forward(&x_2d)?;

        // Reshape back to [batch, seq, d_model]
        let q = q_2d.reshape((batch, seq, self.d_model))?;
        let k = k_2d.reshape((batch, seq, self.d_model))?;
        let v = v_2d.reshape((batch, seq, self.d_model))?;

        // Step 2: Split into heads
        // [batch, num_heads, seq, head_dim]
        let q = self.reshape_to_heads(&q, batch, seq)?;
        let k = self.reshape_to_heads(&k, batch, seq)?;
        let v = self.reshape_to_heads(&v, batch, seq)?;

        // Step 3: Compute attention scores
        // scores = Q @ K^T / √d_k
        // Q: [batch, h, seq, d_head], K^T: [batch, h, d_head, seq]
        // scores: [batch, h, seq, seq]
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q.matmul(&k_t)?.affine(self.scale, 0.0)?;

        // Step 4: Apply causal mask (optional)
        let scores = if self.causal {
            let mask = self.create_causal_mask(seq, x.dtype(), x.device())?;
            // mask is [seq, seq], scores are [batch, h, seq, seq]
            // Broadcasting handles the batch and head dimensions
            scores.add(&mask)?
        } else {
            scores
        };

        // Step 5: Softmax over last dimension (key positions)
        let attn_weights = scores.softmax(3)?; // [batch, h, seq, seq]

        // Step 6: Weighted sum of values
        // [batch, h, seq, seq] @ [batch, h, seq, d_head] = [batch, h, seq, d_head]
        let attn_output = attn_weights.matmul(&v)?;

        // Step 7: Concatenate heads
        // [batch, h, seq, d_head] → [batch, seq, d_model]
        let concat = self.reshape_from_heads(&attn_output, batch, seq)?;

        // Step 8: Output projection
        let concat_2d = concat.reshape((batch * seq, self.d_model))?;
        let output_2d = self.w_o.forward(&concat_2d)?;
        output_2d.reshape((batch, seq, self.d_model))
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        let mut params = Vec::new();
        params.extend(self.w_q.parameters());
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor<B>)> {
        let mut named = Vec::new();
        for (k, v) in self.w_q.named_parameters() {
            named.push((format!("w_q.{k}"), v));
        }
        for (k, v) in self.w_k.named_parameters() {
            named.push((format!("w_k.{k}"), v));
        }
        for (k, v) in self.w_v.named_parameters() {
            named.push((format!("w_v.{k}"), v));
        }
        for (k, v) in self.w_o.named_parameters() {
            named.push((format!("w_o.{k}"), v));
        }
        named
    }
}
