//! # shrew-nn
//!
//! Neural network layers, activations, and loss functions for Shrew.
//!
//! Provides reusable building blocks following the [`Module`] trait pattern
//! (similar to PyTorch's `nn.Module`):
//!
//! 1. **Module trait** — every layer implements `forward()`
//! 2. **Linear** — fully connected: `y = xW^T + b`
//! 3. **Embedding** — lookup table for discrete tokens
//! 4. **Dropout** — regularization via random zeroing
//! 5. **Activations** — ReLU, GELU, SiLU, etc. as modules
//! 6. **Loss functions** — MSELoss, CrossEntropyLoss
//!
//! Modules are generic over `Backend` (like `Tensor<B>`), so the same
//! network definition works on CPU, CUDA, or any future backend.

pub mod activation;
pub mod attention;
pub mod batchnorm;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod flatten;
pub mod groupnorm;
pub mod init;
pub mod layernorm;
pub mod linear;
pub mod loss;
pub mod metrics;
pub mod module;
pub mod rmsnorm;
pub mod rnn;
pub mod sequential;
pub mod transformer;

pub use activation::{GeLU, LeakyReLU, Mish, ReLU, SiLU, Sigmoid, Tanh, ELU};
pub use attention::MultiHeadAttention;
pub use batchnorm::BatchNorm2d;
pub use conv::{AdaptiveAvgPool2d, AvgPool2d, Conv1d, Conv2d, MaxPool2d};
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use flatten::Flatten;
pub use groupnorm::GroupNorm;
pub use layernorm::LayerNorm;
pub use linear::Linear;
pub use loss::{
    bce_loss, bce_with_logits_loss, cross_entropy_loss, l1_loss, l1_loss_with_reduction, mse_loss,
    mse_loss_with_reduction, nll_loss, smooth_l1_loss, Reduction,
};
pub use metrics::{
    accuracy, argmax_classes, classification_report, f1_score, mae, mape, perplexity,
    perplexity_from_log_probs, precision, r2_score, recall, rmse, tensor_accuracy, top_k_accuracy,
    Average, ClassMetrics, ConfusionMatrix,
};
pub use module::Module;
pub use rmsnorm::RMSNorm;
pub use rnn::{GRUCell, LSTMCell, RNNCell, GRU, LSTM, RNN};
pub use sequential::Sequential;
pub use transformer::TransformerBlock;
