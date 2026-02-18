// RAdam — Rectified Adam
//
// RAdam (Liyuan Liu et al., 2019) addresses Adam's variance problem in early
// training. Standard Adam's adaptive learning rate can have high variance in
// the first few steps because the 2nd moment estimate (v) is poorly calibrated.
//
// RAdam automatically detects when the variance of the adaptive learning rate
// is too high and falls back to SGD with momentum until the 2nd moment
// estimate has accumulated enough samples.
//
// The key insight: compute ρ (rho), an approximation of the length of the
// SMA (simple moving average) of the adaptive learning rate. When ρ > 5,
// the variance is low enough to use the adaptive step. Otherwise, use
// a momentum-only step.
//
// This eliminates the need for a learning rate warmup, which is one of the
// most fragile hyperparameters in transformer training.
//
// Update rule:
//   m = β1 * m + (1 - β1) * grad
//   v = β2 * v + (1 - β2) * grad²
//   m_hat = m / (1 - β1^t)
//
//   ρ_inf = 2/(1-β2) - 1
//   ρ_t = ρ_inf - 2*t*β2^t/(1-β2^t)
//
//   if ρ_t > 5:  (variance is tractable → use adaptive step)
//     v_hat = v / (1 - β2^t)
//     r = √((ρ_t-4)(ρ_t-2)ρ_inf / ((ρ_inf-4)(ρ_inf-2)ρ_t))
//     θ = θ - lr * r * m_hat / (√v_hat + ε)
//   else:  (variance too high → momentum-only step)
//     θ = θ - lr * m_hat
//
// HYPERPARAMETERS (same as Adam):
//   lr = 1e-3, β1 = 0.9, β2 = 0.999, ε = 1e-8

use shrew_core::backend::Backend;
use shrew_core::backprop::GradStore;
use shrew_core::error::Result;
use shrew_core::tensor::Tensor;

use crate::optimizer::{Optimizer, OptimizerState, Stateful};

/// Rectified Adam (RAdam) optimizer.
///
/// Automatically switches between adaptive and momentum-only updates
/// based on the variance of the adaptive learning rate, eliminating
/// the need for learning rate warmup.
pub struct RAdam<B: Backend> {
    params: Vec<Tensor<B>>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    /// Step counter
    t: u64,
    /// First moment (mean of gradients)
    m: Vec<Vec<f64>>,
    /// Second moment (mean of squared gradients)
    v: Vec<Vec<f64>>,
    /// ρ_inf = 2/(1-β2) - 1 (precomputed)
    rho_inf: f64,
}

impl<B: Backend> RAdam<B> {
    /// Create a new RAdam optimizer.
    pub fn new(params: Vec<Tensor<B>>, lr: f64) -> Self {
        let n = params.len();
        let beta2 = 0.999;
        RAdam {
            params,
            lr,
            beta1: 0.9,
            beta2,
            epsilon: 1e-8,
            weight_decay: 0.0,
            t: 0,
            m: vec![Vec::new(); n],
            v: vec![Vec::new(); n],
            rho_inf: 2.0 / (1.0 - beta2) - 1.0,
        }
    }

    /// Set β1.
    pub fn beta1(mut self, beta1: f64) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Set β2 (also recomputes ρ_inf).
    pub fn beta2(mut self, beta2: f64) -> Self {
        self.beta2 = beta2;
        self.rho_inf = 2.0 / (1.0 - beta2) - 1.0;
        self
    }

    /// Set ε.
    pub fn epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    /// Set weight decay.
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

    /// Get the step count.
    pub fn step_count(&self) -> u64 {
        self.t
    }
}

impl<B: Backend> Optimizer<B> for RAdam<B> {
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

            let grad_data = grad.to_f64_vec()?;
            let mut param_data = param.to_f64_vec()?;
            let n = param_data.len();

            // Initialize moments on first step
            if self.m[i].is_empty() {
                self.m[i] = vec![0.0; n];
                self.v[i] = vec![0.0; n];
            }

            // Weight decay (decoupled, like AdamW)
            if self.weight_decay != 0.0 {
                for j in 0..n {
                    param_data[j] -= self.lr * self.weight_decay * param_data[j];
                }
            }

            // Update moments
            for (m, &g) in self.m[i].iter_mut().zip(grad_data.iter()) {
                *m = self.beta1 * *m + (1.0 - self.beta1) * g;
            }
            for (v, &g) in self.v[i].iter_mut().zip(grad_data.iter()) {
                *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
            }

            // Bias-corrected first moment
            let bc1 = 1.0 - self.beta1.powi(self.t as i32);

            // Compute ρ_t
            let beta2_t = self.beta2.powi(self.t as i32);
            let bc2 = 1.0 - beta2_t;
            let rho_t = self.rho_inf - 2.0 * self.t as f64 * beta2_t / bc2;

            if rho_t > 5.0 {
                // Variance is tractable: use adaptive step with rectification
                let r = ((rho_t - 4.0) * (rho_t - 2.0) * self.rho_inf
                    / ((self.rho_inf - 4.0) * (self.rho_inf - 2.0) * rho_t))
                    .sqrt();

                for j in 0..n {
                    let m_hat = self.m[i][j] / bc1;
                    let v_hat = self.v[i][j] / bc2;
                    param_data[j] -= self.lr * r * m_hat / (v_hat.sqrt() + self.epsilon);
                }
            } else {
                // Variance too high: use momentum-only step (no adaptive LR)
                for j in 0..n {
                    let m_hat = self.m[i][j] / bc1;
                    param_data[j] -= self.lr * m_hat;
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

impl<B: Backend> Stateful for RAdam<B> {
    fn state_dict(&self) -> OptimizerState {
        let mut state = OptimizerState::new("RAdam");

        state.set_scalar("t", self.t as f64);
        state.set_scalar("lr", self.lr);
        state.set_scalar("beta1", self.beta1);
        state.set_scalar("beta2", self.beta2);
        state.set_scalar("epsilon", self.epsilon);
        state.set_scalar("weight_decay", self.weight_decay);
        state.set_scalar("rho_inf", self.rho_inf);
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
        if state.optimizer_type != "RAdam" {
            return Err(shrew_core::Error::msg(format!(
                "Cannot load {} state into RAdam optimizer",
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
        if let Some(ri) = state.get_scalar("rho_inf") {
            self.rho_inf = ri;
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
