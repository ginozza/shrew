// Adam / AdamW — Adaptive Moment Estimation
//
// Adam is the most widely-used optimizer in deep learning. It maintains
// TWO moving averages per parameter:
//
//   m (1st moment): exponential average of gradients (direction)
//   v (2nd moment): exponential average of squared gradients (magnitude)
//
// Update rule (bias-corrected):
//   m_hat = m / (1 - β1^t)
//   v_hat = v / (1 - β2^t)
//   θ = θ - lr * m_hat / (√v_hat + ε)
//
// WHY ADAM WORKS SO WELL:
//
// - The 1st moment (m) acts like momentum, smoothing gradient direction
// - The 2nd moment (v) acts like per-parameter learning rate scaling:
//   parameters with historically large gradients get smaller updates,
//   and parameters with small gradients get larger updates.
//   This is "adaptive" — hence the name.
//
// AdamW DIFFERENCE:
//
// Standard Adam applies weight decay INSIDE the gradient (coupled).
// AdamW applies weight decay DIRECTLY to the parameters (decoupled).
// This is more principled and is the default for training Transformers.
//
// HYPERPARAMETERS (default values are from the original paper):
//   lr = 1e-3, β1 = 0.9, β2 = 0.999, ε = 1e-8, weight_decay = 0.01

use shrew_core::backend::Backend;
use shrew_core::backprop::GradStore;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use crate::optimizer::{Optimizer, OptimizerState, Stateful};

/// Adam optimizer (Adaptive Moment Estimation).
///
/// Standard defaults: lr=1e-3, β1=0.9, β2=0.999, ε=1e-8
pub struct Adam<B: Backend> {
    params: Vec<Tensor<B>>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    /// Whether to use decoupled weight decay (AdamW style)
    decoupled_decay: bool,
    /// Step counter (for bias correction)
    t: u64,
    /// First moment vectors (one per parameter)
    m: Vec<Vec<f64>>,
    /// Second moment vectors (one per parameter)
    v: Vec<Vec<f64>>,
}

impl<B: Backend> Adam<B> {
    /// Create a standard Adam optimizer.
    pub fn new(params: Vec<Tensor<B>>, lr: f64) -> Self {
        let n = params.len();
        Adam {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            decoupled_decay: false,
            t: 0,
            m: vec![Vec::new(); n],
            v: vec![Vec::new(); n],
        }
    }

    /// Set β1 (1st moment decay rate).
    pub fn beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set β2 (2nd moment decay rate).
    pub fn beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Set ε (numerical stability term).
    pub fn epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    /// Set weight decay (L2 penalty).
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Access current parameters.
    pub fn params(&self) -> &[Tensor<B>] {
        &self.params
    }

    /// Mutable access to current parameters (for checkpoint loading).
    pub fn params_mut(&mut self) -> &mut Vec<Tensor<B>> {
        &mut self.params
    }

    /// Get the step count.
    pub fn step_count(&self) -> u64 {
        self.t
    }

    /// Set the learning rate (used by LR schedulers).
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }
}

/// AdamW optimizer (Adam with decoupled weight decay).
///
/// This is the standard optimizer for training Transformers.
/// The key difference from Adam: weight decay is applied directly to
/// the parameters, not mixed into the gradient.
pub struct AdamW<B: Backend>(pub Adam<B>);

impl<B: Backend> AdamW<B> {
    /// Create an AdamW optimizer with standard defaults.
    ///
    /// Default: lr=1e-3, β1=0.9, β2=0.999, ε=1e-8, weight_decay=0.01
    pub fn new(params: Vec<Tensor<B>>, lr: f64, weight_decay: f64) -> Self {
        let mut adam = Adam::new(params, lr);
        adam.decoupled_decay = true;
        adam.weight_decay = weight_decay;
        AdamW(adam)
    }

    /// Set weight decay.
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.0.weight_decay = wd;
        self
    }

    /// Set β1.
    pub fn beta1(mut self, beta1: f64) -> Self {
        self.0.beta1 = beta1;
        self
    }

    /// Set β2.
    pub fn beta2(mut self, beta2: f64) -> Self {
        self.0.beta2 = beta2;
        self
    }

    /// Access current parameters.
    pub fn params(&self) -> &[Tensor<B>] {
        self.0.params()
    }

    /// Mutable access to current parameters (for checkpoint loading).
    pub fn params_mut(&mut self) -> &mut Vec<Tensor<B>> {
        self.0.params_mut()
    }

    /// Set the learning rate (used by LR schedulers).
    pub fn set_lr(&mut self, lr: f64) {
        self.0.set_lr(lr);
    }
}

impl<B: Backend> Optimizer<B> for Adam<B> {
    #[allow(clippy::needless_range_loop)]
    fn step(&mut self, grads: &GradStore<B>) -> Result<Vec<Tensor<B>>> {
        self.t += 1;
        let mut new_params = Vec::with_capacity(self.params.len());

        for (i, param) in self.params.iter().enumerate() {
            let grad = match grads.get(param) {
                Some(g) => g,
                None => {
                    new_params.push(param.clone());
                    continue;
                }
            };

            let mut grad_data = grad.to_f64_vec()?;
            let mut param_data = param.to_f64_vec()?;
            let n = param_data.len();

            // Initialize moment vectors on first step
            if self.m[i].is_empty() {
                self.m[i] = vec![0.0; n];
                self.v[i] = vec![0.0; n];
            }

            // Coupled weight decay (standard Adam): add to gradient
            if self.weight_decay != 0.0 && !self.decoupled_decay {
                for (g, &p) in grad_data.iter_mut().zip(param_data.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            // Update 1st moment: m = β1 * m + (1 - β1) * grad
            for (m, &g) in self.m[i].iter_mut().zip(grad_data.iter()) {
                *m = self.beta1 * *m + (1.0 - self.beta1) * g;
            }

            // Update 2nd moment: v = β2 * v + (1 - β2) * grad²
            for (v, &g) in self.v[i].iter_mut().zip(grad_data.iter()) {
                *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
            }

            // Bias correction factors
            let bc1 = 1.0 - self.beta1.powi(self.t as i32);
            let bc2 = 1.0 - self.beta2.powi(self.t as i32);

            // Update parameters: θ = θ - lr * m_hat / (√v_hat + ε)
            for j in 0..n {
                let m_hat = self.m[i][j] / bc1;
                let v_hat = self.v[i][j] / bc2;

                // Decoupled weight decay (AdamW): subtract directly from param
                if self.weight_decay != 0.0 && self.decoupled_decay {
                    param_data[j] -= self.lr * self.weight_decay * param_data[j];
                }

                param_data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
            }

            // Update storage in-place so model layers sharing this tensor see new values
            param.update_data_inplace(&param_data)?;

            new_params.push(param.clone());
        }

        // Keep self.params pointing to the same tensors (they've been updated in-place)
        Ok(new_params)
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

impl<B: Backend> Optimizer<B> for AdamW<B> {
    fn step(&mut self, grads: &GradStore<B>) -> Result<Vec<Tensor<B>>> {
        self.0.step(grads)
    }

    fn learning_rate(&self) -> f64 {
        self.0.learning_rate()
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.0.set_learning_rate(lr);
    }
}

// Stateful — Save/restore optimizer internal state

impl<B: Backend> Stateful for Adam<B> {
    fn state_dict(&self) -> OptimizerState {
        let name = if self.decoupled_decay {
            "AdamW"
        } else {
            "Adam"
        };
        let mut state = OptimizerState::new(name);

        state.set_scalar("t", self.t as f64);
        state.set_scalar("lr", self.lr);
        state.set_scalar("beta1", self.beta1);
        state.set_scalar("beta2", self.beta2);
        state.set_scalar("epsilon", self.epsilon);
        state.set_scalar("weight_decay", self.weight_decay);
        state.set_scalar(
            "decoupled_decay",
            if self.decoupled_decay { 1.0 } else { 0.0 },
        );
        state.set_scalar("n_params", self.m.len() as f64);

        for (i, m) in self.m.iter().enumerate() {
            state.set_buffer(format!("m.{i}"), m.clone());
        }
        for (i, v) in self.v.iter().enumerate() {
            state.set_buffer(format!("v.{i}"), v.clone());
        }

        state
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<()> {
        if state.optimizer_type != "Adam" && state.optimizer_type != "AdamW" {
            return Err(shrew_core::Error::msg(format!(
                "Cannot load {} state into Adam/AdamW optimizer",
                state.optimizer_type
            )));
        }

        self.t = state.get_scalar("t").unwrap_or(0.0) as u64;
        if let Some(lr) = state.get_scalar("lr") {
            self.lr = lr;
        }
        if let Some(b1) = state.get_scalar("beta1") {
            self.beta1 = b1;
        }
        if let Some(b2) = state.get_scalar("beta2") {
            self.beta2 = b2;
        }
        if let Some(eps) = state.get_scalar("epsilon") {
            self.epsilon = eps;
        }
        if let Some(wd) = state.get_scalar("weight_decay") {
            self.weight_decay = wd;
        }
        if let Some(dd) = state.get_scalar("decoupled_decay") {
            self.decoupled_decay = dd != 0.0;
        }

        let n = self.m.len();
        for i in 0..n {
            if let Some(buf) = state.get_buffer(&format!("m.{i}")) {
                self.m[i] = buf.clone();
            }
            if let Some(buf) = state.get_buffer(&format!("v.{i}")) {
                self.v[i] = buf.clone();
            }
        }

        Ok(())
    }
}

impl<B: Backend> Stateful for AdamW<B> {
    fn state_dict(&self) -> OptimizerState {
        self.0.state_dict()
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<()> {
        self.0.load_state_dict(state)
    }
}
