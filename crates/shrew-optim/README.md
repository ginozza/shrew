# shrew-optim

Optimizers for gradient-based training in Shrew.

Implements parameter update strategies using gradients computed by autograd:
- **SGD** — Stochastic Gradient Descent (with optional momentum and weight decay)
- **Adam / AdamW** — Adaptive Moment Estimation (decoupled weight decay variant)
- **RAdam** — Rectified Adam (variance-adaptive, no warmup needed)
- **RMSProp** — Root Mean Square Propagation

Also includes:
- **LR Schedulers** — StepLR, ExponentialLR, LinearLR, CosineAnnealingLR, CosineWarmupLR, ReduceLROnPlateau
- **Gradient Clipping** — clip_grad_norm, clip_grad_value, grad_norm
- **EMA** — Exponential Moving Average of model parameters
- **GradAccumulator** — Gradient accumulation over multiple steps

## License

Apache-2.0
