// RMSProp — Root Mean Square Propagation
//
// RMSProp was proposed by Geoff Hinton (in his Coursera lectures, unpublished).
// It maintains a running average of squared gradients to normalize the gradient,
// effectively giving each parameter its own adaptive learning rate.
//
// Update rule:
//   v = α * v + (1 - α) * grad²          (running average of squared gradients)
//   θ = θ - lr * grad / (√v + ε)
//
// With momentum:
//   v = α * v + (1 - α) * grad²
//   buf = momentum * buf + grad / (√v + ε)
//   θ = θ - lr * buf
//
// RMSProp is particularly useful for recurrent neural networks and
// non-stationary objectives. It can be viewed as a precursor to Adam:
// Adam combines RMSProp's adaptive scaling with momentum.
//
// HYPERPARAMETERS:
//   lr = 1e-2 (typical), α = 0.99, ε = 1e-8, momentum = 0, weight_decay = 0

use shrew_core::backend::Backend;
use shrew_core::backprop::GradStore;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use crate::optimizer::{Optimizer, OptimizerState, Stateful};

/// RMSProp optimizer.
///
/// Adapts the learning rate per-parameter using a running average
/// of squared gradients.
///
/// # Default hyperparameters
/// - `lr`: 0.01
/// - `alpha`: 0.99 (smoothing constant)
/// - `epsilon`: 1e-8
/// - `momentum`: 0.0
/// - `weight_decay`: 0.0
pub struct RMSProp<B: Backend> {
    params: Vec<Tensor<B>>,
    lr: f64,
    alpha: f64,
    epsilon: f64,
    momentum: f64,
    weight_decay: f64,
    /// Running average of squared gradients (one per parameter)
    v: Vec<Vec<f64>>,
    /// Momentum buffer (one per parameter, only if momentum > 0)
    buf: Vec<Vec<f64>>,
}

impl<B: Backend> RMSProp<B> {
    /// Create a new RMSProp optimizer with default hyperparameters.
    pub fn new(params: Vec<Tensor<B>>, lr: f64) -> Self {
        let n = params.len();
        RMSProp {
            params,
            lr,
            alpha: 0.99,
            epsilon: 1e-8,
            momentum: 0.0,
            weight_decay: 0.0,
            v: vec![Vec::new(); n],
            buf: vec![Vec::new(); n],
        }
    }

    /// Set the smoothing constant α (default: 0.99).
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set ε (numerical stability, default: 1e-8).
    pub fn epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    /// Set momentum factor (default: 0).
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set weight decay / L2 penalty (default: 0).
    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Access current parameters.
    pub fn params(&self) -> &[Tensor<B>] {
        &self.params
    }

    /// Mutable access to current parameters.
    pub fn params_mut(&mut self) -> &mut Vec<Tensor<B>> {
        &mut self.params
    }
}

impl<B: Backend> Optimizer<B> for RMSProp<B> {
    fn step(&mut self, grads: &GradStore<B>) -> Result<Vec<Tensor<B>>> {
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

            // Initialize state on first step
            if self.v[i].is_empty() {
                self.v[i] = vec![0.0; n];
                if self.momentum > 0.0 {
                    self.buf[i] = vec![0.0; n];
                }
            }

            // Weight decay: add L2 penalty to gradient
            if self.weight_decay != 0.0 {
                for (g, &p) in grad_data.iter_mut().zip(param_data.iter()) {
                    *g += self.weight_decay * p;
                }
            }

            // Update running average of squared gradients: v = α * v + (1 - α) * g²
            for (v, &g) in self.v[i].iter_mut().zip(grad_data.iter()) {
                *v = self.alpha * *v + (1.0 - self.alpha) * g * g;
            }

            if self.momentum > 0.0 {
                // With momentum:
                //   buf = momentum * buf + grad / (√v + ε)
                //   param = param - lr * buf
                for j in 0..n {
                    self.buf[i][j] = self.momentum * self.buf[i][j]
                        + grad_data[j] / (self.v[i][j].sqrt() + self.epsilon);
                    param_data[j] -= self.lr * self.buf[i][j];
                }
            } else {
                // Without momentum:
                //   param = param - lr * grad / (√v + ε)
                for j in 0..n {
                    param_data[j] -= self.lr * grad_data[j] / (self.v[i][j].sqrt() + self.epsilon);
                }
            }

            param.update_data_inplace(&param_data)?;
            new_params.push(param.clone());
        }

        Ok(new_params)
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}

// Stateful — Save/restore optimizer internal state

impl<B: Backend> Stateful for RMSProp<B> {
    fn state_dict(&self) -> OptimizerState {
        let mut state = OptimizerState::new("RMSProp");

        state.set_scalar("lr", self.lr);
        state.set_scalar("alpha", self.alpha);
        state.set_scalar("epsilon", self.epsilon);
        state.set_scalar("momentum", self.momentum);
        state.set_scalar("weight_decay", self.weight_decay);
        state.set_scalar("n_params", self.v.len() as f64);

        for (i, v) in self.v.iter().enumerate() {
            state.set_buffer(format!("v.{i}"), v.clone());
        }
        for (i, b) in self.buf.iter().enumerate() {
            if !b.is_empty() {
                state.set_buffer(format!("buf.{i}"), b.clone());
            }
        }

        state
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> Result<()> {
        if state.optimizer_type != "RMSProp" {
            return Err(shrew_core::Error::msg(format!(
                "Cannot load {} state into RMSProp optimizer",
                state.optimizer_type
            )));
        }

        if let Some(lr) = state.get_scalar("lr") {
            self.lr = lr;
        }
        if let Some(a) = state.get_scalar("alpha") {
            self.alpha = a;
        }
        if let Some(eps) = state.get_scalar("epsilon") {
            self.epsilon = eps;
        }
        if let Some(m) = state.get_scalar("momentum") {
            self.momentum = m;
        }
        if let Some(wd) = state.get_scalar("weight_decay") {
            self.weight_decay = wd;
        }

        let n = self.v.len();
        for i in 0..n {
            if let Some(buf) = state.get_buffer(&format!("v.{i}")) {
                self.v[i] = buf.clone();
            }
            if let Some(buf) = state.get_buffer(&format!("buf.{i}")) {
                self.buf[i] = buf.clone();
            }
        }

        Ok(())
    }
}
