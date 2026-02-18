//! # shrew-optim
//!
//! Optimizers for gradient-based training.
//!
//! Optimizers update model parameters using gradients computed by `backward()`.
//! The training loop is:
//!
//! 1. `output = model.forward(input)`
//! 2. `loss = loss_fn(output, target)`
//! 3. `grads = loss.backward()` — autograd computes gradients
//! 4. `optimizer.step(&grads)` — optimizer updates parameters
//!
//! Implemented optimizers:
//! - **SGD**: Stochastic Gradient Descent (with optional momentum)
//! - **Adam**: Adaptive Moment Estimation
//! - **AdamW**: Adam with decoupled weight decay
//! - **RMSProp**: Root Mean Square Propagation
//! - **RAdam**: Rectified Adam (no warmup needed)

pub mod adam;
pub mod clip;
pub mod ema;
pub mod optimizer;
pub mod radam;
pub mod rmsprop;
pub mod scheduler;
pub mod sgd;

pub use adam::{Adam, AdamW};
pub use clip::{clip_grad_norm, clip_grad_value, grad_norm, GradAccumulator};
pub use ema::EMA;
pub use optimizer::{Optimizer, OptimizerState, Stateful};
pub use radam::RAdam;
pub use rmsprop::RMSProp;
pub use scheduler::{
    CosineAnnealingLR, CosineWarmupLR, ExponentialLR, LinearLR, LrScheduler, ReduceLROnPlateau,
    StepLR,
};
pub use sgd::SGD;
