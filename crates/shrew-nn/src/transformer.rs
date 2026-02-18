// TransformerBlock — One layer of the Transformer architecture
//
// A TransformerBlock is the fundamental building block of models like
// GPT, BERT, LLaMA, and virtually all modern language models.
//
// ARCHITECTURE (Pre-Norm style, used in GPT-2, LLaMA, etc.):
//
//   ┌───────────────────────────────┐
//   │ Input: x [batch, seq, d_model]│
//   └───────────────┬───────────────┘
//                   │
//          ┌────────┴────────┐
//          │   LayerNorm 1   │
//          │   ↓             │
//          │   MHA           │  ← Multi-Head Self-Attention
//          │   ↓             │
//          └────────┬────────┘
//                   │ + x        ← Residual connection
//                   │
//          ┌────────┴────────┐
//          │   LayerNorm 2   │
//          │   ↓             │
//          │   FFN           │  ← Feed-Forward Network
//          │   ↓             │
//          └────────┬────────┘
//                   │ + x        ← Residual connection
//                   │
//   ┌───────────────┴───────────────┐
//   │ Output [batch, seq, d_model]  │
//   └───────────────────────────────┘
//
// FEED-FORWARD NETWORK (FFN):
//
//   FFN(x) = Linear2(GELU(Linear1(x)))
//   Linear1: d_model → d_ff (expand, typically 4×d_model)
//   Linear2: d_ff → d_model (compress back)
//
// The FFN processes each position independently (same Transform applied to
// every token). It's where much of the "knowledge" is stored in the model.
//
// RESIDUAL CONNECTIONS:
//
//   output = x + sublayer(x)
//
// These residual connections are crucial for training deep networks:
// - They provide gradient highways (gradients flow directly through the + x)
// - They allow the sublayer to learn a "correction" rather than the full mapping
// - Without them, transformers with >6 layers would be very hard to train
//
// WHY PRE-NORM?
//
// Pre-norm (LayerNorm before attention/FFN) is more stable for training than
// post-norm (LayerNorm after). Most modern models use pre-norm.

use shrew_core::backend::Backend;
use shrew_core::dtype::DType;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use crate::attention::MultiHeadAttention;
use crate::layernorm::LayerNorm;
use crate::linear::Linear;
use crate::module::Module;

/// A single Transformer block (pre-norm style).
///
/// Contains:
/// - Self-attention with multi-head attention
/// - Feed-forward network (two linear layers with GELU)
/// - Two layer normalizations
/// - Residual connections around both sub-layers
pub struct TransformerBlock<B: Backend> {
    /// LayerNorm before attention
    ln1: LayerNorm<B>,
    /// Multi-head self-attention
    attn: MultiHeadAttention<B>,
    /// LayerNorm before FFN
    ln2: LayerNorm<B>,
    /// FFN first linear: d_model → d_ff
    ff1: Linear<B>,
    /// FFN second linear: d_ff → d_model
    ff2: Linear<B>,
    /// Model dimension
    d_model: usize,
}

impl<B: Backend> TransformerBlock<B> {
    /// Create a new TransformerBlock.
    ///
    /// # Arguments
    /// - `d_model`: model dimension (embedding size)
    /// - `num_heads`: number of attention heads
    /// - `d_ff`: feed-forward inner dimension (typically 4 * d_model)
    /// - `causal`: whether to use causal (autoregressive) attention mask
    /// - `dtype`: data type
    /// - `device`: compute device
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        causal: bool,
        dtype: DType,
        device: &B::Device,
    ) -> Result<Self> {
        let ln1 = LayerNorm::new(d_model, 1e-5, dtype, device)?;
        let attn = MultiHeadAttention::new(d_model, num_heads, dtype, device)?.with_causal(causal);
        let ln2 = LayerNorm::new(d_model, 1e-5, dtype, device)?;
        let ff1 = Linear::new(d_model, d_ff, true, dtype, device)?;
        let ff2 = Linear::new(d_ff, d_model, true, dtype, device)?;

        Ok(TransformerBlock {
            ln1,
            attn,
            ln2,
            ff1,
            ff2,
            d_model,
        })
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Forward pass through the FFN: Linear → GELU → Linear
    fn ffn(&self, x: &Tensor<B>, batch: usize, seq: usize) -> Result<Tensor<B>> {
        // Reshape [batch, seq, d_model] → [batch*seq, d_model] for Linear
        let x_2d = x.reshape((batch * seq, self.d_model))?;
        let h = self.ff1.forward(&x_2d)?.gelu()?;
        // h is [batch*seq, d_ff] — reshape back for ff2
        let out = self.ff2.forward(&h)?;
        out.reshape((batch, seq, self.d_model))
    }
}

impl<B: Backend> Module<B> for TransformerBlock<B> {
    /// Forward pass (pre-norm):
    ///   x = x + attention(layernorm(x))
    ///   x = x + ffn(layernorm(x))
    fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        let dims = x.dims();
        if dims.len() != 3 {
            return Err(shrew_core::Error::msg(format!(
                "TransformerBlock expects [batch, seq, d_model], got {:?}",
                dims
            )));
        }
        let batch = dims[0];
        let seq = dims[1];

        // Sub-layer 1: Self-attention with residual
        let normed1 = self.ln1.forward(x)?;
        let attn_out = self.attn.forward(&normed1)?;
        let x = x.add(&attn_out)?; // Residual connection

        // Sub-layer 2: FFN with residual
        let normed2 = self.ln2.forward(&x)?;
        let ffn_out = self.ffn(&normed2, batch, seq)?;
        x.add(&ffn_out) // Residual connection
    }

    fn parameters(&self) -> Vec<Tensor<B>> {
        let mut params = Vec::new();
        params.extend(self.ln1.parameters());
        params.extend(self.attn.parameters());
        params.extend(self.ln2.parameters());
        params.extend(self.ff1.parameters());
        params.extend(self.ff2.parameters());
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor<B>)> {
        let mut named = Vec::new();
        for (k, v) in self.ln1.named_parameters() {
            named.push((format!("ln1.{k}"), v));
        }
        for (k, v) in self.attn.named_parameters() {
            named.push((format!("attn.{k}"), v));
        }
        for (k, v) in self.ln2.named_parameters() {
            named.push((format!("ln2.{k}"), v));
        }
        for (k, v) in self.ff1.named_parameters() {
            named.push((format!("ff1.{k}"), v));
        }
        for (k, v) in self.ff2.named_parameters() {
            named.push((format!("ff2.{k}"), v));
        }
        named
    }
}
